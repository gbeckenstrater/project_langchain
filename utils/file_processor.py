import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import PyPDF2
import docx
from pathlib import Path
import time
from urllib.parse import urlparse, urljoin
import re

class FileProcessor:
    """Handles reading and processing of various file types and web content"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']
        self.web_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def read_source(self, source: str) -> str:
        """Read content from file path or URL"""
        if self._is_url(source):
            return self._read_webpage(source)
        else:
            return self.read_file(source)
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _read_webpage(self, url: str) -> str:
        """Read content from a webpage"""
        try:
            print(f"🌐 Fetching webpage: {url}")
            
            # Make request with headers to avoid bot detection
            response = requests.get(url, headers=self.web_headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get page title
            title = soup.find('title')
            if title:
                text = f"Title: {title.get_text()}\n\n{text}"
            
            print(f"✅ Successfully extracted {len(text)} characters from webpage")
            return text
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching webpage: {e}")
        except Exception as e:
            raise Exception(f"Error processing webpage: {e}")
    
    def read_file(self, file_path: str) -> str:
        """Read content from various file formats"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.txt' or extension == '.md':
            return self._read_text_file(file_path)
        elif extension == '.pdf':
            return self._read_pdf_file(file_path)
        elif extension == '.docx':
            return self._read_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _read_text_file(self, file_path: str) -> str:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _read_pdf_file(self, file_path: str) -> str:
        """Read PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx_file(self, file_path: str) -> str:
        """Read Word document"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def get_files_in_directory(self, directory: str) -> List[str]:
        """Get all supported files in a directory"""
        files = []
        for file_path in Path(directory).glob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                files.append(str(file_path))
        return files
    
    def chunk_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Split text into chunks for processing"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - overlap
            if start >= text_length:
                break
        
        return chunks
    
    def get_webpage_metadata(self, url: str) -> Dict[str, str]:
        """Extract metadata from webpage"""
        try:
            response = requests.get(url, headers=self.web_headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'url': url,
                'title': soup.find('title').get_text() if soup.find('title') else 'No title',
                'description': '',
                'keywords': '',
                'author': '',
                'domain': urlparse(url).netloc
            }
            
            # Extract meta description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata['description'] = desc_tag.get('content', '')
            
            # Extract meta keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag:
                metadata['keywords'] = keywords_tag.get('content', '')
            
            # Extract author
            author_tag = soup.find('meta', attrs={'name': 'author'})
            if author_tag:
                metadata['author'] = author_tag.get('content', '')
            
            return metadata
            
        except Exception as e:
            return {'url': url, 'error': str(e), 'domain': urlparse(url).netloc}
