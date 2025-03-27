import os
import sys


os.environ["ENVIRONMENT"] = "alembic"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import Base
from app.db.models.base import BaseModel
from app.db.models.user import User
from app.db.models.candidate import Candidate
from app.db.models.chat import Chat, ChatMessage
from app.db.models.knowledge import KnowledgeDocument

target_metadata = Base.metadata