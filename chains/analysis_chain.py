from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from models.data_models import StructuredDocument, AnalysisResult
from config import Config
import json
import re

class DocumentAnalysisChain:
    def __init__(self):
        self.llm = Ollama(
            model=Config.ANALYSIS_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3
        )
        
        self.analysis_prompt = PromptTemplate(
            input_variables=["structured_data"],
            template="""
            You are a strategic analyst. Analyze the following structured document data and provide insights.
            
            Structured Document Data:
            {structured_data}
            
            Provide a comprehensive analysis including:
            1. Key insights (3-5 main takeaways)
            2. Overall sentiment score (-1.0 to 1.0, where -1 is very negative, 0 is neutral, 1 is very positive)
            3. Complexity score (1-10, where 1 is very simple, 10 is very complex)
            4. Actionable recommendations (3-5 recommendations)
            5. Risk factors (potential risks identified)
            6. Opportunities (potential opportunities identified)
            
            IMPORTANT: Return ONLY a valid JSON object with this exact structure. Do not include any other text:
            {{
                "key_insights": ["insight1", "insight2", "insight3"],
                "sentiment_score": 0.5,
                "complexity_score": 7,
                "recommendations": ["rec1", "rec2", "rec3"],
                "risk_factors": ["risk1", "risk2"],
                "opportunities": ["opp1", "opp2"]
            }}
            
            Be analytical, objective, and provide actionable insights.
            """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt,
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
    
    def analyze(self, structured_doc: StructuredDocument) -> AnalysisResult:
        """Analyze structured document and return insights"""
        try:
            # Convert structured doc to JSON for analysis
            doc_json = structured_doc.model_dump_json(indent=2)
            
            print("🤖 Sending analysis request to Ollama...")
            result = self.chain.run(structured_data=doc_json)
            
            print(f"📝 Raw Analysis Response (first 200 chars): {result[:200]}...")
            
            # Try to extract JSON from response
            parsed_data = self.extract_json_from_response(result)
            
            if parsed_data is None:
                print("❌ Could not extract valid JSON from analysis response")
                print(f"🔍 Full response: {result}")
                raise ValueError("No valid JSON found in analysis response")
            
            print("✅ Successfully parsed analysis JSON response")
            
            # Convert to Pydantic model with validation
            analysis_result = AnalysisResult(
                key_insights=parsed_data.get("key_insights", ["Analysis completed"]),
                sentiment_score=max(-1.0, min(1.0, parsed_data.get("sentiment_score", 0.0))),
                complexity_score=max(1, min(10, parsed_data.get("complexity_score", 5))),
                recommendations=parsed_data.get("recommendations", ["Review findings"]),
                risk_factors=parsed_data.get("risk_factors", []),
                opportunities=parsed_data.get("opportunities", [])
            )
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            print("🔧 Creating fallback analysis...")
            
            # Fallback analysis based on document content
            return self.create_fallback_analysis(structured_doc)
    
    def create_fallback_analysis(self, structured_doc: StructuredDocument) -> AnalysisResult:
        """Create a basic analysis when LLM fails"""
        
        # Basic analysis based on document content
        insights = []
        recommendations = []
        
        if "protein" in structured_doc.title.lower():
            insights.append("Document focuses on protein research and optimization")
            recommendations.append("Consider implementing protein intake guidelines")
        
        if len(structured_doc.entities) > 0:
            insights.append(f"Document contains {len(structured_doc.entities)} key entities")
            
        if len(structured_doc.facts) > 0:
            insights.append(f"Document presents {len(structured_doc.facts)} important facts")
        else:
            insights.append("Document structure was successfully extracted")
            
        # Determine complexity based on content
        complexity = 5
        if len(structured_doc.topics) > 3:
            complexity += 2
        if len(structured_doc.entities) > 10:
            complexity += 1
            
        return AnalysisResult(
            key_insights=insights if insights else ["Document analysis completed"],
            sentiment_score=0.1,  # Slightly positive for research content
            complexity_score=min(10, complexity),
            recommendations=recommendations if recommendations else ["Review document content", "Consider follow-up research"],
            risk_factors=["Analysis may be incomplete due to technical limitations"],
            opportunities=["Further analysis with improved tools", "Manual review for detailed insights"]
        )