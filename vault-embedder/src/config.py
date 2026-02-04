"""Configuration handling for vault-embedder."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class VaultConfig:
    """Configuration for vault indexing."""
    
    # Required
    vault_path: str
    collection_name: str
    
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # Embedding model
    model_name: str = "taylorai/bge-micro-v2"
    model_dimensions: int = 384
    
    # File filtering
    include_extensions: list[str] = field(default_factory=lambda: [".md"])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        ".*",           # hidden files/folders
        "_*",           # underscore prefixed
        "node_modules", # common excludes
    ])
    
    # Chunking
    chunk_size: int = 1000       # max characters per chunk
    chunk_overlap: int = 200     # overlap between chunks
    min_chunk_size: int = 100    # minimum chunk size to index
    
    # State tracking
    state_file: Optional[str] = None  # defaults to .vault-embedder-state.json in vault
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "VaultConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "VaultConfig":
        """Create config from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vault_path": self.vault_path,
            "collection_name": self.collection_name,
            "qdrant_url": self.qdrant_url,
            "qdrant_api_key": self.qdrant_api_key,
            "model_name": self.model_name,
            "model_dimensions": self.model_dimensions,
            "include_extensions": self.include_extensions,
            "exclude_patterns": self.exclude_patterns,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "state_file": self.state_file,
        }
    
    @property
    def vault(self) -> Path:
        """Get vault path as Path object."""
        return Path(self.vault_path).expanduser().resolve()
    
    @property
    def state_path(self) -> Path:
        """Get state file path."""
        if self.state_file:
            return Path(self.state_file).expanduser().resolve()
        return self.vault / ".vault-embedder-state.json"


def load_config(config_path: Optional[str] = None) -> VaultConfig:
    """Load configuration from file or environment.
    
    Priority:
    1. Explicit config_path argument
    2. VAULT_EMBEDDER_CONFIG environment variable
    3. ./config.yaml
    4. ~/.config/vault-embedder/config.yaml
    """
    import os
    
    if config_path:
        return VaultConfig.from_yaml(config_path)
    
    env_path = os.environ.get("VAULT_EMBEDDER_CONFIG")
    if env_path and Path(env_path).exists():
        return VaultConfig.from_yaml(env_path)
    
    local_config = Path("./config.yaml")
    if local_config.exists():
        return VaultConfig.from_yaml(local_config)
    
    user_config = Path.home() / ".config" / "vault-embedder" / "config.yaml"
    if user_config.exists():
        return VaultConfig.from_yaml(user_config)
    
    raise FileNotFoundError(
        "No configuration found. Create config.yaml or set VAULT_EMBEDDER_CONFIG"
    )
