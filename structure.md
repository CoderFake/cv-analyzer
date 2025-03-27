    cv-analyzer/
    ├── alembic/                        # Thư mục quản lý migrations cho database
    │   ├── versions/                   # Các file migrations
    │   ├── env.py                      # Cấu hình môi trường Alembic
    │   ├── README                      # Tài liệu Alembic
    │   └── script.py.mako              # Template cho migration script
    │
    ├── app/                            # Thư mục chính chứa code ứng dụng
    │   ├── api/                        # API endpoints
    │   │   ├── __init__.py
    │   │   ├── auth.py                 # API xác thực người dùng
    │   │   ├── candidates.py           # API quản lý CV ứng viên
    │   │   ├── chat.py                 # API chat với ứng viên
    │   │   ├── dependencies.py         # Các dependency cho FastAPI
    │   │   ├── endpoints.py            # Tập hợp tất cả endpoints
    │   │   ├── files.py                # API quản lý files (upload/download)
    │   │   ├── knowledge.py            # API quản lý knowledge base
    │   │   └── knowledge_chat.py       # API chat với knowledge base
    │   │
    │   ├── core/                       # Các module cốt lõi của ứng dụng
    │   │   ├── __init__.py
    │   │   ├── config.py               # Cấu hình ứng dụng (Pydantic Settings)
    │   │   ├── database.py             # Kết nối và quản lý database
    │   │   ├── middleware.py           # Middleware cho FastAPI
    │   │   └── security.py             # Xác thực và bảo mật
    │   │
    │   ├── db/                         # Database models và repositories
    │   │   ├── __init__.py
    │   │   ├── models/                 # SQLAlchemy models
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py             # Base model
    │   │   │   ├── candidate.py        # Model ứng viên
    │   │   │   ├── chat.py             # Model chat và messages
    │   │   │   ├── knowledge.py        # Model knowledge base
    │   │   │   └── user.py             # Model người dùng
    │   │   │
    │   │   └── repositories/           # Các repository tương tác với database
    │   │       ├── __init__.py
    │   │       ├── base.py             # Base repository
    │   │       ├── candidate_repository.py
    │   │       ├── chat_repository.py
    │   │       ├── knowledge_repository.py
    │   │       └── user_repository.py
    │   │
    │   ├── lib/                        # Thư viện và các tích hợp bên thứ ba
    │   │   ├── __init__.py
    │   │   └── llm_web_search/         # Thư viện tìm kiếm web cho LLM
    │   │       ├── __init__.py
    │   │       ├── chunkers/           # Chia nhỏ văn bản thành chunks
    │   │       │   ├── __init__.py
    │   │       │   ├── base_chunker.py
    │   │       │   ├── character_chunker.py
    │   │       │   └── semantic_chunker.py
    │   │       ├── retrievers/         # Lấy thông tin từ chunks
    │   │       │   ├── __init__.py
    │   │       │   ├── bm25_retriever.py
    │   │       │   ├── faiss_retriever.py
    │   │       │   └── splade_retriever.py
    │   │       ├── llm_web_search.py   # Module chính cho web search
    │   │       ├── retrieval.py        # Truy xuất và xử lý kết quả tìm kiếm
    │   │       └── utils.py            # Các tiện ích hỗ trợ
    │   │
    │   ├── schemas/                    # Pydantic schemas cho validation và serialization
    │   │   ├── __init__.py
    │   │   ├── candidate.py            # Schemas cho Candidate
    │   │   ├── chat.py                 # Schemas cho Chat
    │   │   ├── common.py               # Schemas dùng chung
    │   │   ├── knowledge.py            # Schemas cho Knowledge
    │   │   └── user.py                 # Schemas cho User
    │   │
    │   ├── services/                   # Business logic của ứng dụng
    │   │   ├── __init__.py
    │   │   ├── cv_analyzer.py          # Service phân tích CV
    │   │   ├── llm_service.py          # Service tương tác với LLM
    │   │   ├── llm_knowledge_service.py # Service tích hợp LLM với knowledge base
    │   │   ├── storage_service.py      # Service lưu trữ file (R2)
    │   │   ├── web_search_service.py   # Service tìm kiếm web
    │   │   └── web_search_adapter.py   # Adapter cho LLM_Web_search
    │   │
    │   └── utils/                      # Các tiện ích
    │       ├── __init__.py
    │       ├── cv_parser.py            # Phân tích nội dung CV
    │       ├── download_model.py       # Script tải mô hình LLM
    │       └── file_processor.py       # Xử lý file upload
    │
    ├── docker/                         # Cấu hình Docker
    │   ├── app/
    │   │   └── entrypoint.sh           # Script khởi động container FastAPI
    │   └── db/
    │       └── init-scripts/
    │           └── init-database.sh    # Script khởi tạo database
    │
    ├── tests/                          # Unit tests và integration tests
    │   ├── __init__.py
    │   ├── conftest.py                 # Cấu hình pytest
    │   ├── api/                        # Tests cho API endpoints
    │   ├── services/                   # Tests cho services
    │   └── utils/                      # Tests cho utilities
    │
    ├── .env.example                    # File biến môi trường mẫu
    ├── .env                            # Biến môi trường cho development
    ├── .gitignore                      # Danh sách files/folders bỏ qua khi commit
    ├── alembic.ini                     # Cấu hình Alembic
    ├── docker-compose.yml              # Cấu hình Docker Compose
    ├── Dockerfile                      # Cấu hình Docker
    ├── main.py                         # Entry point của ứng dụng
    ├── pyproject.toml                  # Cấu hình Python project
    ├── README.md                       # Tài liệu giới thiệu 
    ├── requirements.txt                # Các dependency Python
    └── structure.md                    # Tài liệu mô tả cấu trúc project