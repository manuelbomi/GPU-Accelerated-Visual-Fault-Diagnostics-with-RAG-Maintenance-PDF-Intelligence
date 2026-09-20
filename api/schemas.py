from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    image_search_ready: bool
    rag_ready: bool
    classifier_ready: bool


class SimilarImageResult(BaseModel):
    name: str
    score: float
    url: str


class SearchResponse(BaseModel):
    query_filename: str
    results: list[SimilarImageResult]


class RelatedImage(BaseModel):
    name: str
    url: str


class RetrievedPassage(BaseModel):
    text: str
    score: float


class DiagnoseResponse(BaseModel):
    query_filename: str
    predicted_label: Optional[str] = None
    predicted_confidence: Optional[float] = None
    top_images: list[RelatedImage]
    related_docs: list[str]
    retrieved_passages: list[RetrievedPassage]
    synthesis: Optional[str] = None
