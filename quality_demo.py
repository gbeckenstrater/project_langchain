#!/usr/bin/env python3
"""
Demo script showing the data quality validation system
"""

from chains.data_quality_agent import DataQualityAgent, QualityScore
from chains.extraction_chain import EnhancedDocumentExtractionChain
from models.data_models import StructuredDocument, ExtractedEntity, ExtractedFact
from rich.console import Console
from rich.panel import Panel

console = Console()

def create_sample_good_document():
    """Create a sample document with good quality data"""
    return StructuredDocument(
        title="Protein Research: Effects of Leucine on Muscle Protein Synthesis",
        summary="This study examines how leucine supplementation affects muscle protein synthesis rates in healthy adults. The research shows significant improvements in MPS following leucine intake.",
        entities=[
            ExtractedEntity(name="leucine", type="compound", context="amino acid supplement", confidence=0.95),
            ExtractedEntity(name="muscle protein synthesis", type="concept", context="biological process", confidence=0.92),
            ExtractedEntity(name="Dr. John Smith", type="person", context="lead researcher", confidence=0.88),
            ExtractedEntity(name="University of Health Sciences", type="organization", context="research institution", confidence=0.90)
        ],
        facts=[
            ExtractedFact(statement="Leucine increases muscle protein synthesis by 25%", category="research finding", importance=9, source_section="results"),
            ExtractedFact(statement="Study included 60 healthy participants", category="methodology", importance=7, source_section="methods"),
            ExtractedFact(statement="Supplementation period was 12 weeks", category="methodology", importance=6, source_section="methods")
        ],
        topics=["protein research", "leucine supplementation", "muscle synthesis", "nutrition science"],
        document_type="research article"
    )

def create_sample_poor_document():
    """Create a sample document with poor quality data"""
    return StructuredDocument(
        title="",  # Missing title
        summary="Short.",  # Too brief
        entities=[
            ExtractedEntity(name="x", type="", context="", confidence=0.1),  # Poor entity
            ExtractedEntity(name="", type="person", context="somewhere", confidence=0.3)  # Empty name
        ],
        facts=[
            ExtractedFact(statement="Bad", category="", importance=15, source_section=""),  # Invalid importance score
            ExtractedFact(statement="", category="test", importance=5, source_section="test")  # Empty statement
        ],
        topics=[],  # No topics
        document_type="unknown"
    )

def demo_quality_validation():
    """Demonstrate the quality validation system"""
    
    console.print(Panel.fit(
        "[bold blue]🔍 Data Quality Validation Demo[/bold blue]\n"
        "[green]Testing good vs poor quality data extraction[/green]",
        border_style="blue"
    ))
    
    quality_agent = DataQualityAgent()
    
    # Test 1: Good quality document
    console.print("\n[bold cyan]Test 1: High Quality Document[/bold cyan]")
    good_doc = create_sample_good_document()
    
    sample_text = """
    Protein Research: Effects of Leucine on Muscle Protein Synthesis
    
    This comprehensive study, led by Dr. John Smith at the University of Health Sciences,
    examines how leucine supplementation affects muscle protein synthesis rates in healthy adults.
    The research shows significant improvements in MPS following leucine intake.
    
    Our 12-week study with 60 healthy participants demonstrated that leucine increases 
    muscle protein synthesis by 25% compared to placebo group.
    """
    
    quality_result = quality_agent.validate_structured_document(good_doc, sample_text)
    quality_agent.print_quality_report(quality_result, "Good Document Sample")
    
    # Test 2: Poor quality document
    console.print("\n[bold cyan]Test 2: Poor Quality Document[/bold cyan]")
    poor_doc = create_sample_poor_document()
    
    poor_text = "Some random text that doesn't match the extracted data at all."
    
    quality_result = quality_agent.validate_structured_document(poor_doc, poor_text)
    quality_agent.print_quality_report(quality_result, "Poor Document Sample")
    
    # Test 3: Real extraction with quality check
    console.print("\n[bold cyan]Test 3: Live Extraction with Quality Check[/bold cyan]")
    extraction_chain = EnhancedDocumentExtractionChain(enable_quality_check=True)
    
    sample_research_text = """
    The Impact of Artificial Intelligence on Healthcare Delivery
    
    Abstract: This paper examines the transformative effects of AI technologies on modern healthcare systems.
    We analyze implementation challenges, benefits, and future opportunities across various medical specialties.
    
    Introduction:
    Artificial Intelligence (AI) has emerged as a revolutionary force in healthcare delivery. 
    From diagnostic imaging to drug discovery, AI applications are reshaping how medical professionals
    approach patient care and treatment decisions.
    
    Key Findings:
    - AI diagnostic tools show 94% accuracy in medical imaging
    - Implementation costs average $2.3M per hospital system
    - Patient satisfaction scores improved by 18% with AI-assisted care
    - Radiologists report 35% faster diagnosis times using AI support
    
    Authors: Dr. Sarah Johnson (Stanford Medical), Prof. Michael Chen (MIT), Dr. Lisa Rodriguez (Mayo Clinic)
    
    Conclusion:
    The integration of AI in healthcare presents significant opportunities for improving patient outcomes
    while reducing costs and increasing efficiency across medical institutions.
    """
    
    structured_doc, quality_acceptable = extraction_chain.extract_with_quality_check(sample_research_text)
    
    console.print(f"\n[bold green]Extraction completed![/bold green]")
    console.print(f"Quality acceptable: {'✅ Yes' if quality_acceptable else '❌ No'}")
    
    console.print(f"\n[bold yellow]Extracted Data Summary:[/bold yellow]")
    console.print(f"Title: {structured_doc.title}")
    console.print(f"Entities: {len(structured_doc.entities)}")
    console.print(f"Facts: {len(structured_doc.facts)}")
    console.print(f"Topics: {', '.join(structured_doc.topics)}")

if __name__ == "__main__":
    demo_quality_validation()