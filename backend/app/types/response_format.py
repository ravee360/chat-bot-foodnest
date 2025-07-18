from pydantic import BaseModel, Field
from typing import Optional,Any,List,Union, Annotated

class TitleResponse(BaseModel):
    title: Optional[str] = Field(
        default="Nothing",
        description="Generate the most suitable and concise title for the given paragraph."
    )

class VLMResponse(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="A concise and relevant title that summarizes the content of the image."
    )
    description: Optional[str] = Field(
        default=None,
        description="Provide a detailed description of the image, including all visible elements. If the image contains a table, extract and include the data. Mention any visible labels, sections (like 'About'), or notable structures."
    )

class ThemeResponse(BaseModel):
    """Schema for theme identification"""
    theme_summary: str = Field(description="Summary of the identified theme")
    supporting_reference_numbers: List[int] = Field(description="List of reference numbers supporting this theme")

class ReferenceResponse(BaseModel):
    """Schema for reference information"""
    reference_number: int = Field(description="Numerical reference identifier")
    source_doc_id: str = Field(description="Source document identifier")
    file_name: str = Field(description="Name of the source file")

class RAGResponse(BaseModel):
    """Schema for the complete RAG response"""
    answer: str = Field(description="Comprehensive answer to the query with inline citations")
    identified_themes: List[ThemeResponse] = Field(description="List of identified themes")
    references: List[ReferenceResponse] = Field(description="Bibliography of references used")

class SVGResponseFormat(BaseModel):
    "Generating SVG Response Format."
    svg : Optional[str] = Field(default="Nothing", description="The svg code to represent the data")

class DynamicPrompt(BaseModel):
    """"Genration of Dynamic Propt"""
    required : bool = Field(default=False, description="SVG Generation required or not")
    prompt : Optional[str] = Field(default="no Prompt", description="Prompt required to generate the svg")