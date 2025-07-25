#!/usr/bin/env python3
"""
Command Line Interface for LangChain Multi-Source Analysis with Data Quality Validation
Usage examples:
  python cli_analyzer.py --url https://www.bbc.co.uk/sport
  python cli_analyzer.py --file "data/input/document.pdf" --no-quality-check
  python cli_analyzer.py --batch urls.txt --strict-quality
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
from chains.extraction_chain import EnhancedDocumentExtractionChain
from chains.analysis_chain import DocumentAnalysisChain
from chains.data_quality_agent import DataQualityAgent, QualityScore
from models.data_models import StructuredDocument, AnalysisResult

console = Console()

class EnhancedCLIAnalyzer:
    """Enhanced CLI analyzer with data quality validation"""
    
    def __init__(self, enable_quality_check: bool = True, strict_quality: bool = False):
        self.file_processor = FileProcessor()
        self.extraction_chain = EnhancedDocumentExtractionChain(enable_quality_check=enable_quality_check)
        self.analysis_chain = DocumentAnalysisChain()
        self.quality_agent = DataQualityAgent() if enable_quality_check else None
        self.strict_quality = strict_quality  # If True, reject low-quality data
        
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def analyze_source(self, source: str, verbose: bool = True) -> tuple[StructuredDocument, AnalysisResult, bool]:
        """
        Analyze a single source with quality validation
        Returns: (StructuredDocument, AnalysisResult, quality_acceptable)
        """
        
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
            
            # Extract structured data with quality check
            if verbose:
                with console.status("[green]Extracting and validating data..."):
                    structured_doc, quality_ok = self.extraction_chain.extract_with_quality_check(content)
            else:
                structured_doc, quality_ok = self.extraction_chain.extract_with_quality_check(content)
            
            # Check if we should proceed with low quality data
            if not quality_ok and self.strict_quality:
                raise ValueError("Data quality is below acceptable threshold (strict mode enabled)")
            
            # Analyze data (only if quality is acceptable or we're not in strict mode)
            if quality_ok or not self.strict_quality:
                if verbose:
                    with console.status("[green]Analyzing content..."):
                        analysis_result = self.analysis_chain.analyze(structured_doc)
                else:
                    analysis_result = self.analysis_chain.analyze(structured_doc)
            else:
                # Create minimal analysis result for low quality data
                analysis_result = AnalysisResult(
                    key_insights=["⚠️ Data quality issues detected - analysis may be unreliable"],
                    sentiment_score=0.0,
                    complexity_score=1,
                    recommendations=["Manual review recommended due to data quality issues"],
                    risk_factors=["Low data quality may lead to incorrect insights"],
                    opportunities=["Improve data extraction process"]
                )
            
            return structured_doc, analysis_result, quality_ok
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing {source}: {e}[/red]")
            raise
    
    def save_results(self, source: str, structured_doc: StructuredDocument, analysis_result: AnalysisResult, quality_ok: bool = True):
        """Save analysis results with quality metadata"""
        
        # Create safe filename
        if self.file_processor._is_url(source):
            from urllib.parse import urlparse
            parsed = urlparse(source)
            safe_name = f"web_{parsed.netloc}_{parsed.path}".replace('/', '_').replace('.', '_')
            safe_name = safe_name.strip('_')[:50]
        else:
            safe_name = Path(source).stem
        
        output_file = Path(Config.OUTPUT_DIR) / f"{safe_name}_results.json"
        
        # Prepare results data with quality metadata
        results = {
            "source": source,
            "source_type": "url" if self.file_processor._is_url(source) else "file",
            "quality_metadata": {
                "quality_check_enabled": self.quality_agent is not None,
                "quality_acceptable": quality_ok,
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
    
    def display_summary(self, source: str, structured_doc: StructuredDocument, analysis_result: AnalysisResult, quality_ok: bool = True):
        """Display analysis summary with quality indicators"""
        
        source_type = "🌐" if self.file_processor._is_url(source) else "📄"
        quality_emoji = "✅" if quality_ok else "⚠️"
        
        # Main results table
        table = Table(title=f"{source_type} Analysis Summary {quality_emoji}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Source", source)
        table.add_row("Data Quality", "✅ Good" if quality_ok else "⚠️ Issues Detected")
        table.add_row("Title", structured_doc.title)
        table.add_row("Type", structured_doc.document_type)
        table.add_row("Entities", str(len(structured_doc.entities)))
        table.add_row("Facts", str(len(structured_doc.facts)))
        table.add_row("Topics", ", ".join(structured_doc.topics))
        table.add_row("Sentiment", f"{analysis_result.sentiment_score:.2f}")
        table.add_row("Complexity", f"{analysis_result.complexity_score}/10")
        
        console.print(table)
        
        # Quality warning if needed
        if not quality_ok:
            console.print(Panel(
                "⚠️ Data quality issues detected. Results may be unreliable.\n"
                "Consider manual review or re-processing with different parameters.", 
                title="🚨 Quality Warning", 
                border_style="red"
            ))
        
        # Key insights
        if analysis_result.key_insights:
            insights_text = "\n".join([f"• {insight}" for insight in analysis_result.key_insights])
            console.print(Panel(insights_text, title="💡 Key Insights", border_style="green"))
        
        # Recommendations
        if analysis_result.recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in analysis_result.recommendations])
            console.print(Panel(rec_text, title="🎯 Recommendations", border_style="blue"))
    
    def interactive_mode(self):
        """Interactive command-line mode with quality options"""
        
        console.print(Panel.fit(
            "[bold blue]🔗 Enhanced Interactive Analysis Mode[/bold blue]\n"
            "[green]Enter sources to analyze (type 'quit' to exit)[/green]\n"
            "[yellow]Quality checking is enabled by default[/yellow]",
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
                
                # Analyze source with quality check
                structured_doc, analysis_result, quality_ok = self.analyze_source(source)
                
                # Display results
                self.display_summary(source, structured_doc, analysis_result, quality_ok)
                
                # Save results
                output_file = self.save_results(source, structured_doc, analysis_result, quality_ok)
                console.print(f"[green]💾 Saved to: {output_file}[/green]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
    
    def batch_analyze(self, sources: List[str], save_results: bool = True):
        """Analyze multiple sources with quality tracking"""
        
        results = []
        quality_stats = {"good": 0, "poor": 0, "total": 0}
        
        with Progress() as progress:
            task = progress.add_task("[green]Processing sources...", total=len(sources))
            
            for source in sources:
                try:
                    structured_doc, analysis_result, quality_ok = self.analyze_source(source, verbose=False)
                    results.append((source, structured_doc, analysis_result, quality_ok))
                    
                    # Update quality stats
                    quality_stats["total"] += 1
                    if quality_ok:
                        quality_stats["good"] += 1
                    else:
                        quality_stats["poor"] += 1
                    
                    if save_results:
                        self.save_results(source, structured_doc, analysis_result, quality_ok)
                    
                    status = "✅" if quality_ok else "⚠️"
                    console.print(f"{status} {source}")
                    
                except Exception as e:
                    console.print(f"❌ {source}: {e}")
                    quality_stats["total"] += 1
                    quality_stats["poor"] += 1
                
                progress.update(task, advance=1)
        
        # Print quality summary
        if quality_stats["total"] > 0:
            good_pct = (quality_stats["good"] / quality_stats["total"]) * 100
            console.print(Panel(
                f"Quality Summary: {quality_stats['good']}/{quality_stats['total']} sources passed quality checks ({good_pct:.1f}%)",
                title="📊 Batch Quality Report",
                border_style="blue"
            ))
        
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
    """Main CLI function with quality check options"""
    
    parser = argparse.ArgumentParser(
        description="LangChain Multi-Source Analysis Tool with Data Quality Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://www.bbc.co.uk/sport
  %(prog)s --file "document.pdf" --no-quality-check
  %(prog)s --batch sources.txt --strict-quality
  %(prog)s --interactive
  %(prog)s --url https://techcrunch.com --no-save
        """
    )
    
    parser.add_argument('--url', '-u', help='Analyze a webpage URL')
    parser.add_argument('--file', '-f', help='Analyze a local file')
    parser.add_argument('--batch', '-b', help='Batch analyze sources from file (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to file')
    parser.add_argument('--no-quality-check', action='store_true', help='Disable data quality validation')
    parser.add_argument('--strict-quality', action='store_true', help='Reject sources with poor data quality')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize analyzer with quality settings
    enable_quality = not args.no_quality_check
    analyzer = EnhancedCLIAnalyzer(
        enable_quality_check=enable_quality,
        strict_quality=args.strict_quality
    )
    
    if not args.quiet:
        quality_status = "✅ Enabled" if enable_quality else "⚠️ Disabled"
        strict_status = " (Strict Mode)" if args.strict_quality else ""
        
        console.print(Panel.fit(
            f"[bold blue]🔗 Enhanced LangChain Multi-Source Analyzer[/bold blue]\n"
            f"[green]📄 Files • 🌐 Websites • 📊 Analysis[/green]\n"
            f"[yellow]🔍 Data Quality Checks: {quality_status}{strict_status}[/yellow]",
            border_style="blue"
        ))
    
    try:
        if args.interactive:
            analyzer.interactive_mode()
        
        elif args.url:
            # Analyze single URL
            structured_doc, analysis_result, quality_ok = analyzer.analyze_source(args.url, not args.quiet)
            
            if not args.quiet:
                analyzer.display_summary(args.url, structured_doc, analysis_result, quality_ok)
            
            if not args.no_save:
                output_file = analyzer.save_results(args.url, structured_doc, analysis_result, quality_ok)
                if not args.quiet:
                    console.print(f"[green]💾 Results saved to: {output_file}[/green]")
        
        elif args.file:
            # Analyze single file
            if not Path(args.file).exists():
                console.print(f"[red]❌ File not found: {args.file}[/red]")
                return 1
            
            structured_doc, analysis_result, quality_ok = analyzer.analyze_source(args.file, not args.quiet)
            
            if not args.quiet:
                analyzer.display_summary(args.file, structured_doc, analysis_result, quality_ok)
            
            if not args.no_save:
                output_file = analyzer.save_results(args.file, structured_doc, analysis_result, quality_ok)
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
                successful = sum(1 for r in results if len(r) >= 3)
                console.print(f"[green]✅ Completed: {successful} successful analyses[/green]")
        
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