#!/usr/bin/env python3
"""
Command Line Interface for LangChain Multi-Source Analysis
Usage examples:
  python cli_analyzer.py --url https://www.bbc.co.uk/sport
  python cli_analyzer.py --file "data/input/document.pdf"
  python cli_analyzer.py --batch urls.txt
  python cli_analyzer.py --interactive
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from utils.file_processor import FileProcessor
from chains.extraction_chain import DocumentExtractionChain
from chains.analysis_chain import DocumentAnalysisChain
from models.data_models import StructuredDocument, AnalysisResult

console = Console()

class CLIAnalyzer:
    """Command-line interface for document and web analysis"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.extraction_chain = DocumentExtractionChain()
        self.analysis_chain = DocumentAnalysisChain()
        
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def analyze_source(self, source: str, verbose: bool = True) -> tuple[StructuredDocument, AnalysisResult]:
        """Analyze a single source (file or URL)"""
        
        source_type = "URL" if self.file_processor._is_url(source) else "File"
        
        if verbose:
            console.print(f"[blue]📊 Analyzing {source_type}: {source}[/blue]")
        
        try:
            # Read content
            if verbose:
                with console.status(f"[green]Reading {source_type.lower()}..."):
                    content = self.file_processor.read_source(source)
            else:
                content = self.file_processor.read_source(source)
            
            if verbose:
                console.print(f"✅ Extracted {len(content)} characters")
            
            # Extract structured data
            if verbose:
                with console.status("[green]Extracting structured data..."):
                    structured_doc = self.extraction_chain.extract(content)
            else:
                structured_doc = self.extraction_chain.extract(content)
            
            # Analyze data
            if verbose:
                with console.status("[green]Analyzing content..."):
                    analysis_result = self.analysis_chain.analyze(structured_doc)
            else:
                analysis_result = self.analysis_chain.analyze(structured_doc)
            
            return structured_doc, analysis_result
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing {source}: {e}[/red]")
            raise
    
    def save_results(self, source: str, structured_doc: StructuredDocument, analysis_result: AnalysisResult):
        """Save analysis results to JSON file"""
        
        # Create safe filename
        if self.file_processor._is_url(source):
            from urllib.parse import urlparse
            parsed = urlparse(source)
            safe_name = f"web_{parsed.netloc}_{parsed.path}".replace('/', '_').replace('.', '_')
            safe_name = safe_name.strip('_')[:50]
        else:
            safe_name = Path(source).stem
        
        output_file = Path(Config.OUTPUT_DIR) / f"{safe_name}_results.json"
        
        # Prepare results data
        results = {
            "source": source,
            "source_type": "url" if self.file_processor._is_url(source) else "file",
            "analysis_metadata": {
                "processed_at": structured_doc.processed_at.isoformat(),
                "entities_count": len(structured_doc.entities),
                "facts_count": len(structured_doc.facts)
            },
            "structured_document": structured_doc.model_dump(),
            "analysis_result": analysis_result.model_dump()
        }
        
        # Add webpage metadata if URL
        if self.file_processor._is_url(source):
            try:
                metadata = self.file_processor.get_webpage_metadata(source)
                results["webpage_metadata"] = metadata
            except:
                pass
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_file
    
    def display_summary(self, source: str, structured_doc: StructuredDocument, analysis_result: AnalysisResult):
        """Display analysis summary"""
        
        source_type = "🌐" if self.file_processor._is_url(source) else "📄"
        
        # Main results table
        table = Table(title=f"{source_type} Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Source", source)
        table.add_row("Title", structured_doc.title)
        table.add_row("Type", structured_doc.document_type)
        table.add_row("Entities", str(len(structured_doc.entities)))
        table.add_row("Facts", str(len(structured_doc.facts)))
        table.add_row("Topics", ", ".join(structured_doc.topics))
        table.add_row("Sentiment", f"{analysis_result.sentiment_score:.2f}")
        table.add_row("Complexity", f"{analysis_result.complexity_score}/10")
        
        console.print(table)
        
        # Key insights
        if analysis_result.key_insights:
            insights_text = "\n".join([f"• {insight}" for insight in analysis_result.key_insights])
            console.print(Panel(insights_text, title="💡 Key Insights", border_style="green"))
        
        # Recommendations
        if analysis_result.recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in analysis_result.recommendations])
            console.print(Panel(rec_text, title="🎯 Recommendations", border_style="blue"))
    
    def interactive_mode(self):
        """Interactive command-line mode"""
        
        console.print(Panel.fit(
            "[bold blue]🔗 Interactive Analysis Mode[/bold blue]\n"
            "[green]Enter sources to analyze (type 'quit' to exit)[/green]",
            border_style="blue"
        ))
        
        while True:
            try:
                source = console.input("\n[cyan]Enter file path or URL: [/cyan]").strip()
                
                if source.lower() in ['quit', 'exit', 'q']:
                    console.print("[green]👋 Goodbye![/green]")
                    break
                
                if not source:
                    continue
                
                # Analyze source
                structured_doc, analysis_result = self.analyze_source(source)
                
                # Display results
                self.display_summary(source, structured_doc, analysis_result)
                
                # Save results
                output_file = self.save_results(source, structured_doc, analysis_result)
                console.print(f"[green]💾 Saved to: {output_file}[/green]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
    
    def batch_analyze(self, sources: List[str], save_results: bool = True):
        """Analyze multiple sources"""
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[green]Processing sources...", total=len(sources))
            
            for source in sources:
                try:
                    structured_doc, analysis_result = self.analyze_source(source, verbose=False)
                    results.append((source, structured_doc, analysis_result))
                    
                    if save_results:
                        self.save_results(source, structured_doc, analysis_result)
                    
                    console.print(f"✅ {source}")
                    
                except Exception as e:
                    console.print(f"❌ {source}: {e}")
                
                progress.update(task, advance=1)
        
        return results

def load_sources_from_file(file_path: str) -> List[str]:
    """Load sources from a text file (one per line)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
        return sources
    except Exception as e:
        console.print(f"[red]Error reading batch file: {e}[/red]")
        return []

def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="LangChain Multi-Source Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://www.bbc.co.uk/sport
  %(prog)s --file "data/input/document.pdf"
  %(prog)s --batch sources.txt
  %(prog)s --interactive
  %(prog)s --url https://techcrunch.com --no-save
        """
    )
    
    parser.add_argument('--url', '-u', help='Analyze a webpage URL')
    parser.add_argument('--file', '-f', help='Analyze a local file')
    parser.add_argument('--batch', '-b', help='Batch analyze sources from file (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CLIAnalyzer()
    
    if not args.quiet:
        console.print(Panel.fit(
            "[bold blue]🔗 LangChain Multi-Source Analyzer[/bold blue]\n"
            "[green]📄 Files • 🌐 Websites • 📊 Analysis[/green]",
            border_style="blue"
        ))
    
    try:
        if args.interactive:
            analyzer.interactive_mode()
        
        elif args.url:
            # Analyze single URL
            structured_doc, analysis_result = analyzer.analyze_source(args.url, not args.quiet)
            
            if not args.quiet:
                analyzer.display_summary(args.url, structured_doc, analysis_result)
            
            if not args.no_save:
                output_file = analyzer.save_results(args.url, structured_doc, analysis_result)
                if not args.quiet:
                    console.print(f"[green]💾 Results saved to: {output_file}[/green]")
        
        elif args.file:
            # Analyze single file
            if not Path(args.file).exists():
                console.print(f"[red]❌ File not found: {args.file}[/red]")
                return 1
            
            structured_doc, analysis_result = analyzer.analyze_source(args.file, not args.quiet)
            
            if not args.quiet:
                analyzer.display_summary(args.file, structured_doc, analysis_result)
            
            if not args.no_save:
                output_file = analyzer.save_results(args.file, structured_doc, analysis_result)
                if not args.quiet:
                    console.print(f"[green]💾 Results saved to: {output_file}[/green]")
        
        elif args.batch:
            # Batch analyze
            sources = load_sources_from_file(args.batch)
            
            if not sources:
                console.print(f"[red]❌ No sources found in: {args.batch}[/red]")
                return 1
            
            if not args.quiet:
                console.print(f"[blue]📊 Batch analyzing {len(sources)} sources...[/blue]")
            
            results = analyzer.batch_analyze(sources, not args.no_save)
            
            if not args.quiet:
                console.print(f"[green]✅ Completed: {len(results)} successful analyses[/green]")
        
        else:
            # No arguments provided, show help
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
