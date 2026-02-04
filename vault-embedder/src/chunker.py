"""Markdown-aware text chunking."""

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    
    text: str
    heading: str | None = None      # Current heading context
    line_start: int = 0             # Starting line number (1-indexed)
    line_end: int = 0               # Ending line number (1-indexed)
    chunk_index: int = 0            # Index within the file
    
    @property
    def char_count(self) -> int:
        return len(self.text)


class MarkdownChunker:
    """Chunks markdown content respecting structure."""
    
    # Patterns for markdown structure
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'^```', re.MULTILINE)
    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL)
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to emit
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, content: str) -> Iterator[Chunk]:
        """Chunk markdown content.
        
        Args:
            content: Markdown text content
            
        Yields:
            Chunk objects with text and metadata
        """
        # Strip frontmatter
        content = self._strip_frontmatter(content)
        
        # Split into sections by headings
        sections = self._split_by_headings(content)
        
        chunk_index = 0
        for section in sections:
            heading = section.get("heading")
            text = section["text"]
            line_start = section["line_start"]
            
            # Further split large sections
            for sub_chunk in self._split_section(text, line_start):
                if len(sub_chunk["text"].strip()) >= self.min_chunk_size:
                    yield Chunk(
                        text=sub_chunk["text"].strip(),
                        heading=heading,
                        line_start=sub_chunk["line_start"],
                        line_end=sub_chunk["line_end"],
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
    
    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from content."""
        match = self.FRONTMATTER_PATTERN.match(content)
        if match:
            return content[match.end():]
        return content
    
    def _split_by_headings(self, content: str) -> list[dict]:
        """Split content into sections by headings."""
        lines = content.split('\n')
        sections = []
        current_section = {
            "heading": None,
            "text": "",
            "line_start": 1,
        }
        
        for i, line in enumerate(lines, start=1):
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # Save current section if it has content
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "heading": heading_match.group(2).strip(),
                    "text": line + '\n',
                    "line_start": i,
                }
            else:
                current_section["text"] += line + '\n'
        
        # Don't forget the last section
        if current_section["text"].strip():
            sections.append(current_section)
        
        return sections
    
    def _split_section(self, text: str, start_line: int) -> Iterator[dict]:
        """Split a section into chunks respecting code blocks.
        
        Args:
            text: Section text
            start_line: Starting line number
            
        Yields:
            Dict with text, line_start, line_end
        """
        if len(text) <= self.chunk_size:
            lines = text.split('\n')
            yield {
                "text": text,
                "line_start": start_line,
                "line_end": start_line + len(lines) - 1,
            }
            return
        
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = ""
        current_start = start_line
        current_line = start_line
        
        for para in paragraphs:
            para_lines = para.count('\n') + 1
            
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para
                current_line += para_lines
            else:
                # Emit current chunk
                if current_chunk.strip():
                    yield {
                        "text": current_chunk,
                        "line_start": current_start,
                        "line_end": current_line - 1,
                    }
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + para
                current_start = current_line - overlap_text.count('\n')
                current_line += para_lines
        
        # Emit remaining
        if current_chunk.strip():
            yield {
                "text": current_chunk,
                "line_start": current_start,
                "line_end": current_line,
            }
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving code blocks."""
        paragraphs = []
        current = ""
        in_code_block = False
        
        for line in text.split('\n'):
            if line.startswith('```'):
                in_code_block = not in_code_block
                current += line + '\n'
            elif not in_code_block and line.strip() == '':
                if current.strip():
                    paragraphs.append(current)
                current = ""
            else:
                current += line + '\n'
        
        if current.strip():
            paragraphs.append(current)
        
        return paragraphs
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at sentence or paragraph boundary
        overlap_region = text[-self.chunk_overlap:]
        
        # Look for sentence boundary
        for sep in ['. ', '.\n', '\n\n', '\n']:
            idx = overlap_region.find(sep)
            if idx != -1:
                return overlap_region[idx + len(sep):]
        
        return overlap_region
