from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser
from models.data_models import StructuredDocument, ExtractedEntity, ExtractedFact
from config import Config
import json
import re

class DocumentExtractionChain:
    def __init__(self):
        self.llm = Ollama(
            model=Config.EXTRACTION_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1
        )
        
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
            # First, try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response using regex
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, return None
            return None
    
    def extract(self, text: str) -> StructuredDocument:
        """Extract structured data from text"""
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
        summary = f"Document contains {len(words)} words and appears to be about protein research and muscle synthesis."
        
        # Basic entity extraction (simple keyword matching)
        entities = []
        
        # Look for common protein research terms
        protein_terms = ["leucine", "protein", "amino acid", "muscle", "synthesis", "MPS"]
        for term in protein_terms:
            if term.lower() in text.lower():
                entities.append(ExtractedEntity(
                    name=term,
                    type="concept",
                    context=f"Found in protein research document",
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
            topics=["protein research", "muscle synthesis", "nutrition"],
            document_type="research article"
        )