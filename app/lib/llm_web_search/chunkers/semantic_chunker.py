import re
from typing import Dict, List, Literal, Optional, Tuple, cast

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..chunkers.character_chunker import RecursiveCharacterTextSplitter
    from ..chunkers.base_chunker import TextSplitter
    from ..utils import Document
except:
    from chunkers.character_chunker import RecursiveCharacterTextSplitter
    from chunkers.base_chunker import TextSplitter
    from utils import Document


def calculate_cosine_distances(sentence_embeddings) -> np.array:

    sliding_windows = np.lib.stride_tricks.sliding_window_view(sentence_embeddings, 2, axis=0)

    dot_prod = np.prod(sliding_windows, axis=-1).sum(axis=1)

    magnitude_prod = np.prod(np.linalg.norm(sliding_windows, axis=1), axis=1)

    cos_sim = dot_prod / magnitude_prod
    return 1 - cos_sim


BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class BoundedSemanticChunker(TextSplitter):

    def __init__(self, embedding_model: SentenceTransformer, buffer_size: int = 1, add_start_index: bool = False,
                 breakpoint_threshold_type: BreakpointThresholdType = "percentile",
                 breakpoint_threshold_amount: Optional[float] = None, number_of_chunks: Optional[int] = None,
                 max_chunk_size: int = 500, min_chunk_size: int = 4):
        super().__init__(add_start_index=add_start_index)
        self._add_start_index = add_start_index
        self.embedding_model = embedding_model
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_split_regex = re.compile(r"(?<=[.?!])\s+")

        assert self.breakpoint_threshold_type == "percentile", "only breakpoint_threshold_type 'percentile' is currently supported"
        assert self.buffer_size == 1, "combining sentences is not supported yet"

    def _calculate_sentence_distances(
        self, sentences: List[dict]
    ) -> Tuple[List[float], List[dict]]:
        sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        embeddings = self.embedding_model.encode(sentences)
        return calculate_cosine_distances(embeddings.tolist())

    def _calculate_breakpoint_threshold(self, distances: np.array, alt_breakpoint_threshold_amount=None) -> float:
        if alt_breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = self.breakpoint_threshold_amount
        else:
            breakpoint_threshold_amount = alt_breakpoint_threshold_amount
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, breakpoint_threshold_amount),
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + breakpoint_threshold_amount * np.std(distances),
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(distances) + breakpoint_threshold_amount * iqr
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:

        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        sentences = self.sentence_split_regex.split(text)

        if len(sentences) == 1:
            return sentences

        bad_sentences = []

        distances = self._calculate_sentence_distances(sentences)

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(
                distances
            )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        for index in indices_above_thresh:
            end_index = index

            group = sentences[start_index : end_index + 1]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

            start_index = index + 1

        if start_index < len(sentences):
            group = sentences[start_index:]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

        if len(bad_sentences) > 0:
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_size, chunk_overlap=10,
                                                                separators=["\n\n", "\n", ".", ", ", " ", ""])
            for bad_sentence in bad_sentences:
                if len(bad_sentence) >= self.min_chunk_size:
                    chunks.extend(recursive_splitter.split_text(bad_sentence))
        return chunks

