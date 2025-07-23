from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from models.data_models import StructuredDocument, ExtractedEntity, ExtractedFact
from config import Config
import json
import re
from typing import List, Dict, Tuple
from enum import Enum

class QualityScore(Enum):
    """Quality score levels"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD" 
    FAIR = "FAIR"
    POOR = "POOR"
    GARBAGE = "GARBAGE"

class DataQualityIssue:
    """Represents a data quality issue found"""
    def __init__(self, issue_type: str, severity: str, description: str, field: str = None):
        self.issue_type = issue_type
        self.severity = severity  # HIGH, MEDIUM, LOW
        self.description = description
        self.field = field

class DataQualityResult:
    """Result of data quality validation"""
    def __init__(self):
        self.overall_score: QualityScore = QualityScore.GOOD
        self.confidence: float = 0.0
        self.issues: List[DataQualityIssue] = []
        self.passed_checks: int = 0
        self.total_checks: int = 0
        self.recommendations: List[str] = []
        self.is_acceptable: bool = True

class DataQualityAgent:
    """AI Agent for validating data quality of extracted documents"""
    
    def __init__(self):
        self.llm = Ollama(
            model=Config.ANALYSIS_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.1  # Low temperature for consistent validation
        )
        
        self.validation_prompt = PromptTemplate(
            input_variables=["structured_data", "original_text_sample"],
            template="""
            You are a Data Quality Validation Agent. Your job is to assess the quality of extracted structured data.
            
            Original Text Sample (first 500 chars):
            {original_text_sample}
            
            Extracted Structured Data:
            {structured_data}
            
            Evaluate the data quality by checking:
            1. ACCURACY: Do entities and facts match the source text?
            2. COMPLETENESS: Are important elements missing?
            3. CONSISTENCY: Are entity types and categories logical?
            4. RELEVANCE: Is extracted info actually important?
            5. FORMAT: Are confidence scores and importance ratings reasonable?
            
            Rate each aspect (1-5 scale):
            - Accuracy: How well does extracted data match source?
            - Completeness: How much important info was captured?
            - Consistency: Are classifications logical and consistent?
            - Relevance: Is the extracted information actually useful?
            - Format: Are scores and ratings reasonable?
            
            IMPORTANT: Return ONLY a valid JSON object:
            {{
                "overall_quality": "EXCELLENT|GOOD|FAIR|POOR|GARBAGE",
                "accuracy_score": 4,
                "completeness_score": 3,
                "consistency_score": 5,
                "relevance_score": 4,
                "format_score": 4,
                "confidence": 0.85,
                "critical_issues": ["issue1", "issue2"],
                "warnings": ["warning1", "warning2"],
                "recommendations": ["rec1", "rec2"],
                "is_acceptable": true
            }}
            
            Be thorough but fair in your assessment.
            """
        )
        
        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=self.validation_prompt,
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
    
    def validate_structured_document(self, structured_doc: StructuredDocument, original_text: str = "") -> DataQualityResult:
        """Main validation method"""
        print("🔍 Running data quality validation...")
        
        result = DataQualityResult()
        
        # Rule-based checks first
        self._run_rule_based_checks(structured_doc, result)
        
        # AI-powered validation if we have original text
        if original_text:
            ai_result = self._run_ai_validation(structured_doc, original_text)
            if ai_result:
                self._merge_ai_results(result, ai_result)
        
        # Calculate final scores
        self._calculate_final_score(result)
        
        return result
    
    def _run_rule_based_checks(self, doc: StructuredDocument, result: DataQualityResult):
        """Run rule-based validation checks"""
        checks_run = 0
        checks_passed = 0
        
        # Check 1: Title quality
        checks_run += 1
        if doc.title and len(doc.title.strip()) > 5 and doc.title != "Untitled Document":
            checks_passed += 1
        else:
            result.issues.append(DataQualityIssue(
                "TITLE_QUALITY", "MEDIUM", 
                "Title is missing, too short, or generic", "title"
            ))
        
        # Check 2: Summary quality
        checks_run += 1
        if doc.summary and len(doc.summary.split()) >= 10:
            checks_passed += 1
        else:
            result.issues.append(DataQualityIssue(
                "SUMMARY_QUALITY", "MEDIUM",
                "Summary is missing or too brief", "summary"
            ))
        
        # Check 3: Entity validation
        checks_run += 1
        valid_entities = 0
        for entity in doc.entities:
            if (entity.name and len(entity.name.strip()) > 1 and 
                entity.type and entity.confidence >= 0.3):
                valid_entities += 1
        
        if len(doc.entities) > 0 and valid_entities / len(doc.entities) >= 0.7:
            checks_passed += 1
        else:
            result.issues.append(DataQualityIssue(
                "ENTITY_QUALITY", "HIGH",
                f"Only {valid_entities}/{len(doc.entities)} entities meet quality standards", "entities"
            ))
        
        # Check 4: Facts validation
        checks_run += 1
        valid_facts = 0
        for fact in doc.facts:
            if (fact.statement and len(fact.statement.split()) >= 3 and
                1 <= fact.importance <= 10):
                valid_facts += 1
        
        if len(doc.facts) == 0 or valid_facts / len(doc.facts) >= 0.6:
            checks_passed += 1
        else:
            result.issues.append(DataQualityIssue(
                "FACTS_QUALITY", "HIGH",
                f"Only {valid_facts}/{len(doc.facts)} facts meet quality standards", "facts"
            ))
        
        # Check 5: Topics validation
        checks_run += 1
        if doc.topics and len(doc.topics) >= 1 and all(len(topic.strip()) > 2 for topic in doc.topics):
            checks_passed += 1
        else:
            result.issues.append(DataQualityIssue(
                "TOPICS_QUALITY", "LOW",
                "Topics are missing or too generic", "topics"
            ))
        
        result.passed_checks = checks_passed
        result.total_checks = checks_run
    
    def _run_ai_validation(self, doc: StructuredDocument, original_text: str) -> dict:
        """Run AI-powered validation"""
        try:
            doc_json = doc.model_dump_json(indent=2)
            text_sample = original_text[:500] + "..." if len(original_text) > 500 else original_text
            
            print("🤖 Running AI quality validation...")
            response = self.validation_chain.run(
                structured_data=doc_json,
                original_text_sample=text_sample
            )
            
            return self.extract_json_from_response(response)
            
        except Exception as e:
            print(f"⚠️ AI validation failed: {e}")
            return None
    
    def _merge_ai_results(self, result: DataQualityResult, ai_result: dict):
        """Merge AI validation results with rule-based results"""
        if not ai_result:
            return
        
        # Update confidence
        result.confidence = ai_result.get("confidence", 0.5)
        
        # Add AI-identified issues
        for issue in ai_result.get("critical_issues", []):
            result.issues.append(DataQualityIssue(
                "AI_CRITICAL", "HIGH", issue
            ))
        
        for warning in ai_result.get("warnings", []):
            result.issues.append(DataQualityIssue(
                "AI_WARNING", "MEDIUM", warning
            ))
        
        # Add recommendations
        result.recommendations.extend(ai_result.get("recommendations", []))
        
        # Update acceptability
        if not ai_result.get("is_acceptable", True):
            result.is_acceptable = False
    
    def _calculate_final_score(self, result: DataQualityResult):
        """Calculate final quality score"""
        # Base score from rule-based checks
        if result.total_checks > 0:
            rule_score = result.passed_checks / result.total_checks
        else:
            rule_score = 0.5
        
        # Count severity of issues
        high_issues = sum(1 for issue in result.issues if issue.severity == "HIGH")
        medium_issues = sum(1 for issue in result.issues if issue.severity == "MEDIUM")
        
        # Determine overall quality
        if rule_score >= 0.9 and high_issues == 0:
            result.overall_score = QualityScore.EXCELLENT
        elif rule_score >= 0.7 and high_issues <= 1:
            result.overall_score = QualityScore.GOOD
        elif rule_score >= 0.5 and high_issues <= 2:
            result.overall_score = QualityScore.FAIR
        elif rule_score >= 0.3:
            result.overall_score = QualityScore.POOR
        else:
            result.overall_score = QualityScore.GARBAGE
            result.is_acceptable = False
        
        # Set recommendations based on score
        if result.overall_score in [QualityScore.POOR, QualityScore.GARBAGE]:
            result.recommendations.insert(0, "⚠️ RECOMMEND MANUAL REVIEW - Data quality is below acceptable threshold")
            result.is_acceptable = False
    
    def print_quality_report(self, result: DataQualityResult, source: str = ""):
        """Print a formatted quality report"""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Quality score with emoji
        score_emoji = {
            QualityScore.EXCELLENT: "🌟",
            QualityScore.GOOD: "✅", 
            QualityScore.FAIR: "⚠️",
            QualityScore.POOR: "❌",
            QualityScore.GARBAGE: "🗑️"
        }
        
        emoji = score_emoji.get(result.overall_score, "❓")
        
        # Main table
        table = Table(title=f"{emoji} Data Quality Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        if source:
            table.add_row("Source", source)
        table.add_row("Overall Score", f"{emoji} {result.overall_score.value}")
        table.add_row("Confidence", f"{result.confidence:.2f}")
        table.add_row("Checks Passed", f"{result.passed_checks}/{result.total_checks}")
        table.add_row("Acceptable", "✅ Yes" if result.is_acceptable else "❌ No")
        
        console.print(table)
        
        # Issues
        if result.issues:
            issues_text = ""
            for issue in result.issues:
                severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(issue.severity, "•")
                issues_text += f"{severity_emoji} {issue.description}\n"
            
            console.print(Panel(issues_text.strip(), title="🔍 Quality Issues", border_style="red"))
        
        # Recommendations
        if result.recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in result.recommendations])
            console.print(Panel(rec_text, title="💡 Recommendations", border_style="blue"))