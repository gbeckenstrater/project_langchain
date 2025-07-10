from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExtractedEntity(BaseModel):
    """Represents an extracted entity from text"""
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type of entity (person, organization, location, etc.)")
    context: str = Field(description="Context where the entity was found")
    confidence: float = Field(description="Confidence score (0-1)")

class ExtractedFact(BaseModel):
    """Represents an extracted fact or statement"""
    statement: str = Field(description="The factual statement")
    category: str = Field(description="Category of the fact")
    importance: int = Field(description="Importance score (1-10)")
    source_section: str = Field(description="Section where fact was found")

class StructuredDocument(BaseModel):
    """Structured representation of a document"""
    title: str = Field(description="Document title")
    summary: str = Field(description="Brief summary")
    entities: List[ExtractedEntity] = Field(description="Extracted entities")
    facts: List[ExtractedFact] = Field(description="Extracted facts")
    topics: List[str] = Field(description="Main topics covered")
    document_type: str = Field(description="Type of document")
    processed_at: datetime = Field(default_factory=datetime.now)

class AnalysisResult(BaseModel):
    """Result of document analysis"""
    key_insights: List[str] = Field(description="Key insights from analysis")
    sentiment_score: float = Field(description="Overall sentiment (-1 to 1)")
    complexity_score: int = Field(description="Complexity score (1-10)")
    recommendations: List[str] = Field(description="Recommendations based on analysis")
    risk_factors: List[str] = Field(description="Identified risk factors")
    opportunities: List[str] = Field(description="Identified opportunities")