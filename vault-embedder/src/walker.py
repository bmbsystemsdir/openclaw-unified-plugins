"""File system walker with gitignore-style exclusions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import xxhash
import pathspec


@dataclass
class FileInfo:
    """Information about a file to index."""
    
    path: Path                  # Absolute path
    relative_path: str          # Path relative to vault root
    mtime: float               # Modification time
    size: int                  # File size in bytes
    content_hash: str = ""     # Hash of content for change detection
    
    def compute_hash(self) -> str:
        """Compute content hash."""
        content = self.path.read_text(encoding='utf-8', errors='replace')
        self.content_hash = xxhash.xxh64(content.encode()).hexdigest()
        return self.content_hash


@dataclass
class IndexState:
    """Tracks indexed files for incremental updates."""
    
    files: dict[str, dict] = field(default_factory=dict)  # relative_path -> {hash, mtime, chunk_ids}
    collection_name: str = ""
    model_name: str = ""
    last_indexed: Optional[float] = None
    
    @classmethod
    def load(cls, path: Path) -> "IndexState":
        """Load state from file."""
        if not path.exists():
            return cls()
        
        try:
            data = json.loads(path.read_text())
            return cls(
                files=data.get("files", {}),
                collection_name=data.get("collection_name", ""),
                model_name=data.get("model_name", ""),
                last_indexed=data.get("last_indexed"),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()
    
    def save(self, path: Path):
        """Save state to file."""
        import time
        self.last_indexed = time.time()
        
        data = {
            "files": self.files,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "last_indexed": self.last_indexed,
        }
        path.write_text(json.dumps(data, indent=2))
    
    def needs_reindex(self, file_info: FileInfo) -> bool:
        """Check if a file needs to be reindexed."""
        stored = self.files.get(file_info.relative_path)
        if not stored:
            return True
        
        # Check if content hash changed
        if stored.get("hash") != file_info.content_hash:
            return True
        
        return False
    
    def update_file(self, file_info: FileInfo, chunk_ids: list[str]):
        """Update state for a file."""
        self.files[file_info.relative_path] = {
            "hash": file_info.content_hash,
            "mtime": file_info.mtime,
            "chunk_ids": chunk_ids,
        }
    
    def remove_file(self, relative_path: str) -> list[str]:
        """Remove a file from state, returning its chunk IDs."""
        if relative_path in self.files:
            chunk_ids = self.files[relative_path].get("chunk_ids", [])
            del self.files[relative_path]
            return chunk_ids
        return []


class VaultWalker:
    """Walks vault directory finding files to index."""
    
    def __init__(
        self,
        vault_path: Path,
        include_extensions: list[str],
        exclude_patterns: list[str],
    ):
        """Initialize walker.
        
        Args:
            vault_path: Root path of the vault
            include_extensions: File extensions to include (e.g., ['.md'])
            exclude_patterns: Gitignore-style patterns to exclude
        """
        self.vault_path = vault_path.resolve()
        self.include_extensions = set(ext.lower() for ext in include_extensions)
        
        # Build pathspec from exclude patterns
        self.exclude_spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            exclude_patterns
        )
        
        # Also load .gitignore if present
        gitignore_path = self.vault_path / ".gitignore"
        if gitignore_path.exists():
            gitignore_patterns = gitignore_path.read_text().splitlines()
            gitignore_spec = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern,
                gitignore_patterns
            )
            self.exclude_spec = self.exclude_spec + gitignore_spec
    
    def walk(self) -> Iterator[FileInfo]:
        """Walk vault and yield files to index.
        
        Yields:
            FileInfo objects for each file to index
        """
        for path in self.vault_path.rglob("*"):
            if not path.is_file():
                continue
            
            # Check extension
            if path.suffix.lower() not in self.include_extensions:
                continue
            
            # Get relative path for pattern matching
            relative = path.relative_to(self.vault_path)
            relative_str = str(relative)
            
            # Check exclusions
            if self.exclude_spec.match_file(relative_str):
                continue
            
            # Build file info
            stat = path.stat()
            file_info = FileInfo(
                path=path,
                relative_path=relative_str,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
            
            yield file_info
    
    def find_deleted(self, state: IndexState) -> list[str]:
        """Find files that were indexed but no longer exist.
        
        Args:
            state: Current index state
            
        Returns:
            List of relative paths that were deleted
        """
        deleted = []
        for relative_path in state.files:
            full_path = self.vault_path / relative_path
            if not full_path.exists():
                deleted.append(relative_path)
        return deleted
