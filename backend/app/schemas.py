from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


# ---------- auth / profile ----------
class UserCreate(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    password: str = Field(min_length=8, max_length=72)  # bcrypt ignores bytes past 72
    display_name: str | None = Field(default=None, max_length=100)


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str | None = None
    created_at: datetime
    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SubjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class SubjectOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    document_count: int = 0


class ProfileOut(BaseModel):
    user: UserOut
    subject_count: int
    document_count: int
    ready_document_count: int
    chunk_count: int
    concept_count: int = 0
    verified_concept_count: int = 0
    subjects: list[SubjectOut]


# ---------- documents / QA ----------
class DocumentOut(BaseModel):
    id: int
    filename: str
    subject_id: int | None = None
    page_count: int
    chunk_count: int
    status: str
    error: str | None = None
    concept_status: str = "none"      # none | running | done | failed
    concept_progress: int = 0
    concept_error: str | None = None
    created_at: datetime
    model_config = {"from_attributes": True}


class DocumentMove(BaseModel):
    subject_id: int | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_ids: list[int] | None = None
    subject_id: int | None = None
    # auto: my material first, then general knowledge | material: strict | general: no retrieval
    mode: Literal["auto", "material", "general"] = "auto"
    conversation_id: int | None = None   # None = start a new saved chat
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    ref: int
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    similarity: float
    text: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]
    provider: str
    source: str = "material"   # material | general | none
    conversation_id: int | None = None


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[Citation] | None = None
    subject_id: int | None = None
    created_at: datetime
    model_config = {"from_attributes": True}


# ---------- concepts ----------
class PrereqOut(BaseModel):
    id: int
    name: str
    source: str          # llm | manual
    verified: bool
    confidence: float


class ConceptOut(BaseModel):
    id: int
    subject_id: int
    name: str
    description: str | None = None
    source: str
    verified: bool
    mastery: float = 0.3
    n_attempts: int = 0
    chunk_count: int = 0
    prerequisites: list[PrereqOut] = []


class ConceptCreate(BaseModel):
    subject_id: int
    name: str = Field(min_length=2, max_length=80)
    description: str | None = Field(default=None, max_length=500)


class ConceptUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    verified: bool | None = None


class PrereqSet(BaseModel):
    prerequisite_ids: list[int]


class MergeRequest(BaseModel):
    keep_id: int
    merge_ids: list[int] = Field(min_length=1)


class PassageOut(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    text: str


# ---------- concepts ----------
class PrereqEdge(BaseModel):
    id: int            # the prerequisite concept
    name: str
    source: str        # llm | manual
    verified: bool


class ConceptOut(BaseModel):
    id: int
    name: str
    description: str | None = None
    subject_id: int | None = None
    source: str        # llm | manual
    verified: bool
    created_at: datetime
    mastery: float
    n_attempts: int
    chunk_count: int
    dependents: int    # how many concepts require this one
    prerequisites: list[PrereqEdge]


class ConceptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    subject_id: int | None = None


class ConceptUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    verified: bool | None = None


class MergeRequest(BaseModel):
    source_id: int     # this concept is merged INTO target and then removed
    target_id: int


class PrereqSet(BaseModel):
    prerequisite_ids: list[int] = Field(default_factory=list, max_length=50)


class ExtractOut(BaseModel):
    document_id: int
    status: str


# ---------- conversations ----------
class ConversationOut(BaseModel):
    id: int
    title: str
    subject_id: int | None = None
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


# ---------- concepts ----------
class PrereqOut(BaseModel):
    id: int                 # the prerequisite concept's id
    name: str
    source: str             # llm | manual
    verified: bool
    confidence: float


class ConceptOut(BaseModel):
    id: int
    subject_id: int
    name: str
    description: str | None = None
    source: str
    verified: bool
    chunk_count: int = 0
    dependent_count: int = 0          # how many concepts need this one
    prerequisites: list[PrereqOut] = []


class ConceptCreate(BaseModel):
    subject_id: int
    name: str = Field(min_length=2, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class ConceptUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    verified: bool | None = None


class PrereqCreate(BaseModel):
    prerequisite_id: int


class PrereqUpdate(BaseModel):
    verified: bool


class MergeRequest(BaseModel):
    into_id: int


class ExtractRequest(BaseModel):
    subject_id: int
    document_ids: list[int] | None = None
    max_chunks: int = Field(default=60, ge=1, le=300)


class ExtractResult(BaseModel):
    chunks_processed: int
    batches: int
    batches_failed: int
    new_concepts: int
    reused_concepts: int
    new_links: int
    new_prerequisites: int
    provider: str


class SourceChunkOut(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    text: str
