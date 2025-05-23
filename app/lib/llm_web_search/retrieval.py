import re
import asyncio
import warnings
import logging
from typing import List, Dict, Iterable, Callable, Iterator
from collections import defaultdict
from itertools import chain

import aiohttp
import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForMaskedLM
import optimum.bettertransformer.transformation

try:
    from app.lib.llm_web_search.retrievers.faiss_retriever import FaissRetriever
    from app.lib.llm_web_search.retrievers.bm25_retriever import BM25Retriever
    from app.lib.llm_web_search.retrievers.splade_retriever import SpladeRetriever
    from app.lib.llm_web_search.chunkers.semantic_chunker import BoundedSemanticChunker
    from app.lib.llm_web_search.chunkers.character_chunker import RecursiveCharacterTextSplitter
    from app.lib.llm_web_search.utils import Document, MySentenceTransformer
except ImportError:
    from retrievers.faiss_retriever import FaissRetriever
    from retrievers.bm25_retriever import BM25Retriever
    from retrievers.splade_retriever import SpladeRetriever
    from chunkers.semantic_chunker import BoundedSemanticChunker
    from chunkers.character_chunker import RecursiveCharacterTextSplitter
    from utils import Document, MySentenceTransformer


class DocumentRetriever:

    def __init__(self, device="cuda", num_results: int = 5, similarity_threshold: float = 0.5, chunk_size: int = 500,
                 ensemble_weighting: float = 0.5, splade_batch_size: int = 2, keyword_retriever: str = "bm25",
                 model_cache_dir: str = None, chunking_method: str = "character-based",
                 chunker_breakpoint_threshold_amount: int = 10, client_timeout: int = 10):
        self.device = device
        self.embedding_model = MySentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_cache_dir,
                                                     device=device,
                                                     model_kwargs={"torch_dtype": torch.float32 if device == "cpu" else torch.float16})
        if keyword_retriever == "splade":
            splade_kwargs = {"cache_dir": model_cache_dir,
                             "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
                             "attn_implementation":"eager",
                             "use_safetensors": False}  # avoid weird redundant asynchronous safetensors download
                                                        # after main download has already finished
            self.splade_doc_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc",
                                                                      cache_dir=model_cache_dir)
            self.splade_doc_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-doc",
                                                                         **splade_kwargs).to(self.device)
            self.splade_query_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-query",
                                                                        cache_dir=model_cache_dir)
            self.splade_query_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-query",
                                                                           **splade_kwargs).to(self.device)
            optimum_logger = optimum.bettertransformer.transformation.logger
            original_log_level = optimum_logger.level
            # Set the level to 'ERROR' to ignore "The BetterTransformer padding during training warning"
            optimum_logger.setLevel(logging.ERROR)
            self.splade_doc_model.to_bettertransformer()
            self.splade_query_model.to_bettertransformer()
            optimum_logger.setLevel(original_log_level)
            self.splade_batch_size = splade_batch_size

        self.spaces_regex = re.compile(r" {3,}")
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunker_breakpoint_threshold_amount = chunker_breakpoint_threshold_amount
        self.ensemble_weighting = ensemble_weighting
        self.keyword_retriever = keyword_retriever
        self.client_timeout = client_timeout

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " \n")
        text = self.spaces_regex.sub(" ", text)
        text = text.strip()
        return text

    def retrieve_from_snippets(self, query: str, documents: list[Document]) -> list[Document]:
        yield "Retrieving relevant results..."
        faiss_retriever = FaissRetriever(self.embedding_model, num_results=self.num_results,
                                         similarity_threshold=self.similarity_threshold)
        faiss_retriever.add_documents(documents)
        return faiss_retriever.get_relevant_documents(query)

    def retrieve_from_webpages(self, query: str, url_list: list[str]) -> list[Document]:
        if self.chunking_method == "semantic":
            text_splitter = BoundedSemanticChunker(self.embedding_model, breakpoint_threshold_type="percentile",
                                                   breakpoint_threshold_amount=self.chunker_breakpoint_threshold_amount,
                                                   max_chunk_size=self.chunk_size)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=10,
                                                                separators=["\n\n", "\n", ".", ", ", " ", ""])
        yield "Downloading and chunking webpages..."
        split_docs = asyncio.run(async_fetch_chunk_websites(url_list, text_splitter, self.client_timeout))

        yield "Retrieving relevant results..."
        if self.ensemble_weighting > 0:
            faiss_retriever = FaissRetriever(self.embedding_model, num_results=self.num_results,
                                          similarity_threshold=self.similarity_threshold)
            faiss_retriever.add_documents(split_docs)
            dense_result_docs = faiss_retriever.get_relevant_documents(query)
        else:
            dense_result_docs = []

        if self.ensemble_weighting < 1:
            #  The sparse keyword retriever is good at finding relevant documents based on keywords,
            #  while the dense retriever is good at finding relevant documents based on semantic similarity.
            if self.keyword_retriever == "bm25":
                keyword_retriever = BM25Retriever.from_documents(split_docs,
                                                                 preprocess_func=self.preprocess_text)
                keyword_retriever.k = self.num_results
            elif self.keyword_retriever == "splade":
                keyword_retriever = SpladeRetriever(
                    splade_doc_tokenizer=self.splade_doc_tokenizer,
                    splade_doc_model=self.splade_doc_model,
                    splade_query_tokenizer=self.splade_query_tokenizer,
                    splade_query_model=self.splade_query_model,
                    device=self.device,
                    batch_size=self.splade_batch_size,
                    k=self.num_results
                )
                keyword_retriever.add_documents(split_docs)
            else:
                raise ValueError("self.keyword_retriever must be one of ('bm25', 'splade')")
            sparse_results_docs = keyword_retriever.get_relevant_documents(query)
        else:
            sparse_results_docs = []

        return weighted_reciprocal_rank([dense_result_docs, sparse_results_docs],
                                        weights=[self.ensemble_weighting, 1 - self.ensemble_weighting])[:self.num_results]


async def async_download_html(url: str, headers: Dict, timeout: int):
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(timeout),
                                     max_field_size=65536) as session:
        try:
            resp = await session.get(url)
            return await resp.text(), url
        except UnicodeDecodeError:
            if not resp.headers['Content-Type'].startswith("text/html"):
                print(f"LLM_Web_search | {url} generated an exception: Expected content type text/html. Got {resp.headers['Content-Type']}.")
        except TimeoutError:
            print('LLM_Web_search | %r did not load in time' % url)
        except Exception as exc:
            print('LLM_Web_search | %r generated an exception: %s' % (url, exc))
    return None


async def async_fetch_chunk_websites(urls: List[str],
                                     text_splitter: BoundedSemanticChunker or RecursiveCharacterTextSplitter,
                                     timeout: int = 10):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
               "Accept-Encoding": "br;q=1.0, gzip;q=0.8, *;q=0.1"}
    result_futures = [async_download_html(url, headers, timeout) for url in urls]
    chunks = []
    for f in asyncio.as_completed(result_futures):
        result = await f
        if result:
            resp_html, url = result
            document = html_to_plaintext_doc(resp_html, url)
            chunks.extend(text_splitter.split_documents([document]))
    return chunks


def docs_to_pretty_str(docs) -> str:
    ret_str = ""
    for i, doc in enumerate(docs):
        ret_str += f"Result {i+1}:\n"
        ret_str += f"{doc.page_content}\n"
        ret_str += f"Source URL: {doc.metadata['source']}\n\n"
    return ret_str


def download_html(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}

    response = requests.get(url, headers=headers, verify=True, timeout=8)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("text/html"):
        raise ValueError(f"Expected content type text/html. Got {content_type}.")
    return response.content


def html_to_plaintext_doc(html_text: str or bytes, url: str) -> Document:
    with warnings.catch_warnings(action="ignore"):
        soup = BeautifulSoup(html_text, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = '\n'.join([s.strip() for s in soup.stripped_strings])
    webpage_document = Document(page_content=strings, metadata={"source": url})
    return webpage_document


def weighted_reciprocal_rank(doc_lists: List[List[Document]], weights: List[float], c: int = 60) -> List[Document]:

    if len(doc_lists) != len(weights):
        raise ValueError(
            "Number of rank lists must be equal to the number of weights."
        )

    rrf_score: Dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + c)

    all_docs = chain.from_iterable(doc_lists)
    sorted_docs = sorted(
        unique_by_key(all_docs, lambda doc: doc.page_content),
        reverse=True,
        key=lambda doc: rrf_score[doc.page_content],
    )
    return sorted_docs


def unique_by_key(iterable: Iterable, key: Callable) -> Iterator:

    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e
