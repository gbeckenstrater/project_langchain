import os
import json
from pathlib import Path
from typing import List, Union
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from urllib.parse import urlparse

from config import Config
from utils.file_processor import FileProcessor
from chains.extraction_chain import DocumentExtractionChain
from chains.analysis_chain import DocumentAnalysisChain
from models.data_models import StructuredDocument, AnalysisResult

console = Console()

class EnhancedDocumentPipeline:
    """Enhanced E2E document processing pipeline for files and web content"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.extraction_chain = DocumentExtractionChain()
        self.analysis_chain = DocumentAnalysisChain()
        
        # Ensure directories exist
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def process_source(self, source: str) -> tuple[StructuredDocument, AnalysisResult]:
        """Process a file or URL through the entire pipeline"""
        
        source_type = "URL" if self.file_processor._is_url(source) else "File"
        console.print(f"[blue]Processing {source_type}: {source}[/blue]")
        
        # Step 1: Read content
        with console.status(f"[bold green]Reading {source_type.lower()}..."):
            raw_text = self.file_processor.read_source(source)
        
        # Step 2: Extract structured data
        with console.status("[bold green]Extracting structured data..."):
            structured_doc = self.extraction_chain.extract(raw_text)
        
        # Step 3: Analyze structured data
        with console.status("[bold green]Analyzing data..."):
            analysis_result = self.analysis_chain.analyze(structured_doc)
        
        return structured_doc, analysis_result
    
    def process_multiple_sources(self, sources: List[str]) -> List[tuple[str, StructuredDocument, AnalysisResult]]:
        """Process multiple files and/or URLs"""
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[green]Processing sources...", total=len(sources))
            
            for source in sources:
                try:
                    structured_doc, analysis_result = self.process_source(source)
                    results.append((source, structured_doc, analysis_result))
                    
                    # Save results
                    self.save_results(source, structured_doc, analysis_result)
                    
                except Exception as e:
                    console.print(f"[red]Error processing {source}: {e}[/red]")
                
                progress.update(task, advance=1)
        
        return results
    
    def process_directory(self, input_dir: str = None) -> List[tuple[str, StructuredDocument, AnalysisResult]]:
        """Process all files in a directory"""
        
        if input_dir is None:
            input_dir = Config.INPUT_DIR
        
        files = self.file_processor.get_files_in_directory(input_dir)
        
        if not files:
            console.print(f"[red]No supported files found in {input_dir}[/red]")
            return []
        
        return self.process_multiple_sources(files)
    
    def save_results(self, source: str, structured_doc: StructuredDocument, analysis_result: AnalysisResult):
        """Save processing results to output directory"""
        
        # Create safe filename from source
        if self.file_processor._is_url(source):
            # For URLs, use domain and path
            parsed = urlparse(source)
            safe_name = f"{parsed.netloc}_{parsed.path}".replace('/', '_').replace('.', '_')
            safe_name = safe_name.strip('_')[:50]  # Limit length
        else:
            # For files, use filename
            safe_name = Path(source).stem
        
        output_file = Path(Config.OUTPUT_DIR) / f"{safe_name}_results.json"
        
        # Add metadata about source type
        results = {
            "source": source,
            "source_type": "url" if self.file_processor._is_url(source) else "file",
            "structured_document": structured_doc.model_dump(),
            "analysis_result": analysis_result.model_dump()
        }
        
        # Add webpage metadata if it's a URL
        if self.file_processor._is_url(source):
            try:
                metadata = self.file_processor.get_webpage_metadata(source)
                results["webpage_metadata"] = metadata
            except:
                pass
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"[green]Results saved to: {output_file}[/green]")
    
    def display_results(self, results: List[tuple[str, StructuredDocument, AnalysisResult]]):
        """Display processing results in a nice format"""
        
        for source, structured_doc, analysis_result in results:
            source_type = "🌐 URL" if self.file_processor._is_url(source) else "📄 File"
            display_name = source if len(source) < 60 else source[:57] + "..."
            
            # Create summary table
            table = Table(title=f"Analysis Results: {source_type}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Source", display_name)
            table.add_row("Title", structured_doc.title)
            table.add_row("Document Type", structured_doc.document_type)
            table.add_row("Entities Found", str(len(structured_doc.entities)))
            table.add_row("Facts Extracted", str(len(structured_doc.facts)))
            table.add_row("Topics", ", ".join(structured_doc.topics))
            table.add_row("Sentiment Score", f"{analysis_result.sentiment_score:.2f}")
            table.add_row("Complexity Score", str(analysis_result.complexity_score))
            
            console.print(table)
            
            # Display summary
            if structured_doc.summary:
                console.print(Panel(structured_doc.summary, title="📝 Summary", border_style="yellow"))
            
            # Display key insights
            if analysis_result.key_insights:
                insights_text = "\n".join([f"• {insight}" for insight in analysis_result.key_insights])
                console.print(Panel(insights_text, title="💡 Key Insights", border_style="green"))
            
            # Display recommendations
            if analysis_result.recommendations:
                rec_text = "\n".join([f"• {rec}" for rec in analysis_result.recommendations])
                console.print(Panel(rec_text, title="🎯 Recommendations", border_style="blue"))
            
            console.print("\n" + "="*80 + "\n")

def main():
    """Main execution function with interactive options"""
    
    console.print(Panel.fit(
        "[bold blue]🔗 LangChain Multi-Source Analysis Pipeline[/bold blue]\n"
        "[green]📄 PDFs • 🌐 Websites • 📝 Documents[/green]",
        border_style="blue"
    ))
    
    # Initialize pipeline
    pipeline = EnhancedDocumentPipeline()
    
    # Interactive menu
    console.print("\n[bold cyan]Choose analysis mode:[/bold cyan]")
    console.print("1. 📁 Process files from input directory")
    console.print("2. 🌐 Analyze a webpage URL")
    console.print("3. 📝 Analyze specific file")
    console.print("4. 🔄 Batch process multiple sources")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Process directory
        results = pipeline.process_directory()
        
        if not results:
            # Create sample if no files
            console.print(f"[yellow]No files found in {Config.INPUT_DIR}[/yellow]")
            console.print("[yellow]Creating sample file...[/yellow]")
            
            sample_file = Path(Config.INPUT_DIR) / "sample_document.txt"
            sample_content = """
            Tech Industry Analysis - 2024
            
            The technology sector continues to evolve rapidly with AI, cloud computing, and cybersecurity 
            leading innovation. Major companies like Google, Microsoft, and Amazon are investing heavily 
            in artificial intelligence capabilities.
            
            Key Trends:
            - AI integration across all platforms
            - Increased focus on data privacy
            - Remote work technology adoption
            - Sustainable computing initiatives
            
            Market Outlook:
            The tech sector shows strong growth potential despite economic uncertainties. 
            Companies focusing on AI and automation are expected to outperform.
            """
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            
            console.print(f"[green]Created sample file: {sample_file}[/green]")
            results = pipeline.process_directory()
    
    elif choice == "2":
        # Analyze webpage
        url = input("Enter webpage URL (e.g., https://www.bbc.co.uk/sport): ").strip()
        if url:
            try:
                structured_doc, analysis_result = pipeline.process_source(url)
                results = [(url, structured_doc, analysis_result)]
            except Exception as e:
                console.print(f"[red]Error processing URL: {e}[/red]")
                return
        else:
            console.print("[red]No URL provided[/red]")
            return
    
    elif choice == "3":
        # Analyze specific file
        file_path = input("Enter file path: ").strip()
        if file_path and Path(file_path).exists():
            try:
                structured_doc, analysis_result = pipeline.process_source(file_path)
                results = [(file_path, structured_doc, analysis_result)]
            except Exception as e:
                console.print(f"[red]Error processing file: {e}[/red]")
                return
        else:
            console.print("[red]File not found[/red]")
            return
    
    elif choice == "4":
        # Batch process
        console.print("\n[cyan]Enter sources (files or URLs), one per line. Empty line to finish:[/cyan]")
        sources = []
        while True:
            source = input("Source: ").strip()
            if not source:
                break
            sources.append(source)
        
        if sources:
            results = pipeline.process_multiple_sources(sources)
        else:
            console.print("[red]No sources provided[/red]")
            return
    
    else:
        console.print("[red]Invalid choice[/red]")
        return
    
    # Display results
    if results:
        console.print("\n[bold green]✨ Analysis Complete![/bold green]")
        pipeline.display_results(results)
    else:
        console.print("[red]No sources were processed successfully[/red]")

if __name__ == "__main__":
    main()
