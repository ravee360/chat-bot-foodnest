# backend/app/models/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

# --- Processing Mode Enum ---
class ProcessingMode(str, Enum):
    ACCURATE_LLM = "accurate_llm"
    FAST_RULE_BASED = "fast_rule_based" # Kept for potential future use, though not active in UI

# --- Document Processing Schemas ---
class DocumentProcessResponse(BaseModel):
    message: str
    file_name: str
    source_doc_id: Optional[str] = None
    status: str = "processing"
    processing_mode_used: Optional[ProcessingMode] = None

# --- Chat and RAG Schemas ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query.")
    collection: Optional[str] = None
    top_k: Optional[int] = 1  # Number of top results to return (default 1)

class EvidenceSnippet(BaseModel):
    text: str
    page: Optional[int] = None
    paragraph: Optional[int] = None
    source_doc_id: Optional[str] = None

class ThemeObject(BaseModel):
    theme_summary: str
    supporting_reference_numbers: List[int]
    evidence_snippets: Optional[List[EvidenceSnippet]] = None  # Fixed reference to EvidenceSnippet

class ReferenceObject(BaseModel): # NEW: For the bibliography
    """
    Represents a single reference in the bibliography.
    """
    reference_number: int
    source_doc_id: str
    file_name: str

class DocumentDetail(BaseModel):
    source_doc_id: str
    file_name: str
    extracted_answer: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None

class RAGResponse(BaseModel):
    """
    Response model from the RAG service, using numerical citations.
    """
    answer: str
    themes: List[ThemeObject]
    references: List[ReferenceObject]
    retrieved_context_document_ids: List[str]
    synthesized_expert_answer: Optional[str] = None
    document_details: Optional[List[DocumentDetail]] = None  # Added for document-level details
    svg : Optional[str] = None

# --- General Error Schema ---
class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None 
    msg: str                       
    type: Optional[str] = None

class HTTPErrorResponse(BaseModel):
    detail: List[ErrorDetail]


if __name__ == "__main__":
    example_rag_resp = RAGResponse(
        answer="The concept is explained in [1, Page: 5, Para: 2] and further elaborated in [2, Page: 10, Para: 1].",
        themes=[
            ThemeObject(theme_summary="Key Concept A", supporting_reference_numbers=[1, 2]),
            ThemeObject(theme_summary="Related Concept B", supporting_reference_numbers=[2])
        ],
        references=[
            ReferenceObject(reference_number=1, source_doc_id="doc_xyz_123", file_name="paper_alpha.pdf"),
            ReferenceObject(reference_number=2, source_doc_id="doc_abc_789", file_name="paper_beta.pdf")
        ],
        retrieved_context_document_ids=["doc_xyz_123", "doc_abc_789", "doc_another_one"]
    )
    print("Example RAGResponse (with numerical citations):")
    print(example_rag_resp.model_dump_json(indent=2))
