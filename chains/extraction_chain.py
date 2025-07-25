from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from models.data_models import StructuredDocument, ExtractedEntity, ExtractedFact
from chains.data_quality_agent import DataQualityAgent, QualityScore
from config import Config
import json
import re

class EnhancedDocumentExtractionChain:
    """Enhanced extraction chain with built-in data quality validation"""
    
    def __init__(self, enable_quality_check: bool = True):
        self.llm = Ollama(
            model=Config.EXTRACTION_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        self.enable_quality_check = enable_quality_check
        if enable_quality_check:
            self.quality_agent = DataQualityAgent()
        
        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an expert document analyzer. Extract structured information from the following text.
            
            Text to analyze:
            {text}
            
            Please extract:
            1. A clear title for this document
            2. A brief summary (2-3 sentences)
            3. Important entities (people, organizations, locations, etc.)
            4. Key facts and statements
            5. Main topics covered
            6. Document type (report, article, memo, etc.)
            
            For entities, include:
            - name: the entity name
            - type: person, organization, location, concept, etc.
            - context: where/how it appears
            - confidence: 0.0-1.0 score
            
            For facts, include:
            - statement: the factual statement
            - category: financial, technical, legal, etc.
            - importance: 1-10 score
            - source_section: which part of document
            
            IMPORTANT: Return ONLY a valid JSON object with this exact structure. Do not include any other text:
            {{
                "title": "document title",
                "summary": "brief summary",
                "entities": [
                    {{"name": "entity name", "type": "entity type", "context": "context", "confidence": 0.9}}
                ],
                "facts": [
                    {{"statement": "fact", "category": "category", "importance": 8, "source_section": "section"}}
                ],
                "topics": ["topic1", "topic2"],
                "document_type": "type"
            }}
            
            Be precise and thorough. Only extract information that is clearly stated in the text.
            """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.extraction_prompt,
            verbose=True
        )
    
    def extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from potentially messy LLM response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            return None
    
    def extract_with_quality_check(self, text: str, max_retries: int = 2) -> tuple[StructuredDocument, bool]:
        """
        Extract structured data with quality validation
        Returns: (StructuredDocument, is_quality_acceptable)
        """
        attempt = 1
        
        while attempt <= max_retries + 1:
            print(f"🔄 Extraction attempt {attempt}/{max_retries + 1}")
            
            # Perform extraction
            structured_doc = self._perform_extraction(text)
            
            if not self.enable_quality_check:
                return structured_doc, True
            
            # Validate quality
            quality_result = self.quality_agent.validate_structured_document(structured_doc, text)
            
            # Print quality report
            self.quality_agent.print_quality_report(quality_result)
            
            # Check if quality is acceptable
            if quality_result.is_acceptable or attempt > max_retries:
                if not quality_result.is_acceptable:
                    print("⚠️ Proceeding with low-quality data after max retries")
                
                return structured_doc, quality_result.is_acceptable
            
            # If quality is poor, try again with more focused prompt
            print(f"🔄 Quality check failed ({quality_result.overall_score.value}), retrying...")
            attempt += 1
            
            # Could implement retry logic with different prompts here
        
        return structured_doc, False
    
    def extract(self, text: str) -> StructuredDocument:
        """Standard extraction method for backward compatibility"""
        structured_doc, _ = self.extract_with_quality_check(text)
        return structured_doc
    
    def _perform_extraction(self, text: str) -> StructuredDocument:
        """Core extraction logic"""
        try:
            print("🤖 Sending request to Ollama...")
            result = self.chain.run(text=text)
            
            print(f"📝 Raw LLM Response (first 200 chars): {result[:200]}...")
            
            # Try to extract JSON from response
            parsed_data = self.extract_json_from_response(result)
            
            if parsed_data is None:
                print("❌ Could not extract valid JSON from response")
                print(f"🔍 Full response: {result}")
                raise ValueError("No valid JSON found in response")
            
            print("✅ Successfully parsed JSON response")
            
            # Convert to Pydantic model
            entities = []
            for entity_data in parsed_data.get("entities", []):
                try:
                    entities.append(ExtractedEntity(**entity_data))
                except Exception as e:
                    print(f"⚠️ Skipping invalid entity: {entity_data} - {e}")
            
            facts = []
            for fact_data in parsed_data.get("facts", []):
                try:
                    facts.append(ExtractedFact(**fact_data))
                except Exception as e:
                    print(f"⚠️ Skipping invalid fact: {fact_data} - {e}")
            
            structured_doc = StructuredDocument(
                title=parsed_data.get("title", "Untitled Document"),
                summary=parsed_data.get("summary", "No summary available"),
                entities=entities,
                facts=facts,
                topics=parsed_data.get("topics", []),
                document_type=parsed_data.get("document_type", "unknown")
            )
            
            return structured_doc
            
        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            print("🔧 Creating fallback structured document...")
            
            # Fallback: create basic structure from text analysis
            return self.create_fallback_structure(text)
    
    def create_fallback_structure(self, text: str) -> StructuredDocument:
        """Create a basic structure when LLM fails"""
        
        # Simple text analysis fallback
        lines = text.split('\n')
        title = "Extracted Document"
        
        # Try to find a title in first few lines
        for line in lines[:10]:
            if len(line.strip()) > 10 and len(line.strip()) < 100:
                title = line.strip()
                break
        
        # Create basic summary
        words = text.split()
        summary = f"Document contains {len(words)} words and appears to be about {title.lower()}."
        
        # Basic entity extraction (simple keyword matching)
        entities = []
        
        # Look for common terms based on title
        if "protein" in title.lower():
            protein_terms = ["leucine", "protein", "amino acid", "muscle", "synthesis", "MPS"]
            for term in protein_terms:
                if term.lower() in text.lower():
                    entities.append(ExtractedEntity(
                        name=term,
                        type="concept",
                        context=f"Found in document about {title}",
                        confidence=0.8
                    ))
        
        # Look for author names (basic pattern)
        author_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)'
        authors = re.findall(author_pattern, text)
        for author in authors[:5]:  # Limit to first 5
            entities.append(ExtractedEntity(
                name=author,
                type="person",
                context="Potential author or researcher",
                confidence=0.7
            ))
        
        return StructuredDocument(
            title=title,
            summary=summary,
            entities=entities,
            facts=[],
            topics=["general", "document analysis"],
            document_type="document"
        )

# For backward compatibility - alias to the original class name
DocumentExtractionChain = EnhancedDocumentExtractionChain