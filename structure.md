
    cv-analyzer/
    ├── app/
    │   ├── api/
    │   │   ├── __init__.py
    │   │   ├── auth.py                    # API endpoints xác thực
    │   │   ├── candidates.py              # API endpoints quản lý CV ứng viên
    │   │   ├── chat.py                    # API endpoints chat
    │   │   ├── dependencies.py            # Các dependency cho API
    │   │   ├── endpoints.py               # Tổng hợp tất cả endpoints
    │   │   ├── files.py                   # API endpoints upload/download file
    │   │   ├── knowledge.py               # API endpoints quản lý knowledge base
    │   │   └── knowledge_chat.py          # API endpoints chat với knowledge base
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py                  # Cấu hình ứng dụng Pydantic 2.0
    │   │   ├── database.py                # Kết nối database
    │   │   ├── security.py                # Xác thực và bảo mật
    │   │   └── middleware.py              # Middleware cho FastAPI
    │   ├── db/
    │   │   ├── __init__.py                # Import models cho Alembic migrations
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py                # Base repository Pydantic 2.0
    │   │   │   ├── candidate_repository.py
    │   │   │   ├── chat_repository.py
    │   │   │   ├── knowledge_repository.py
    │   │   │   └── user_repository.py
    │   │   └── models/
    │   │       ├── __init__.py
    │   │       ├── base.py                # Base model
    │   │       ├── candidate.py           # Model ứng viên 
    │   │       ├── chat.py                # Model chat và messages
    │   │       ├── knowledge.py           # Model knowledge base
    │   │       └── user.py                # Model người dùng
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── cv_analyzer.py             # Service phân tích CV
    │   │   ├── llm_service.py             # Service LLM cho phân tích và chat
    │   │   ├── llm_knowledge_service.py   # Service LLM tích hợp knowledge base
    │   │   ├── storage_service.py         # Service lưu trữ file (R2)
    │   │   ├── web_search_service.py      # Service tìm kiếm web
    │   │   └── web_search_adapter.py      # Adapter tích hợp LLM_Web_search
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   ├── candidate.py               # Pydantic schemas cho Candidate
    │   │   ├── chat.py                    # Pydantic schemas cho Chat
    │   │   ├── common.py                  # Pydantic schemas chung
    │   │   ├── knowledge.py               # Pydantic schemas cho Knowledge
    │   │   └── user.py                    # Pydantic schemas cho User
    │   └── utils/
    │       ├── __init__.py
    │       ├── cv_parser.py               # Công cụ phân tích CV
    │       ├── file_processor.py          # Xử lý file upload
    │       └── download_model.py          # Script tải mô hình LLM
    ├── alembic/
    │   ├── versions/                      # Các phiên bản migration
    │   ├── env.py                         # Môi trường Alembic
    │   ├── README
    │   └── script.py.mako
    ├── docker/
    │   ├── db/
    │   │   └── init-scripts/
    │   │       └── init-database.sh       # Script khởi tạo database
    │   └── app/
    │       └── entrypoint.sh              # Script khởi động container FastAPI
    ├── .env.example                       # File môi trường mẫu
    ├── .env.dev                           # Môi trường development
    ├── .env.stg                           # Môi trường staging
    ├── .env.prod                          # Môi trường production
    ├── .gitignore
    ├── alembic.ini                        # Cấu hình Alembic
    ├── Dockerfile                         # Cấu hình Docker
    ├── docker-compose.yml                 # Cấu hình Docker Compose
    ├── main.py                            # Entry point của ứng dụng
    ├── README.md                          # Tài liệu hướng dẫn
    └── requirements.txt                   # Các dependency Python