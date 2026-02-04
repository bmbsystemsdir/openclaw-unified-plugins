# vault-embedder

Local vault indexer using `taylorai/bge-micro-v2` embeddings with Qdrant vector storage.

## Overview

Indexes markdown files from Obsidian vaults (or any markdown directory) into Qdrant for semantic search. Designed for use with OpenClaw agents.

**Key features:**
- Local embeddings using `bge-micro-v2` (384 dimensions, ~24MB model)
- Stores in Qdrant with separate collections per vault
- Markdown-aware chunking (respects headers, code blocks)
- Incremental updates (only re-embeds changed files)
- Gitignore-style exclusions

## Architecture

```
Vault (markdown files)
    │
    ▼
┌─────────────────┐
│  File Walker    │ ── respects .gitignore, config exclusions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunker       │ ── markdown-aware splitting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  bge-micro-v2   │ ── local embedding model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Qdrant       │ ── vector storage + search
└─────────────────┘
```

## Collections

| Agent  | Collection       | Vault Path              |
|--------|------------------|-------------------------|
| Steve  | `vault_work`     | `/home/steve/obsidian-vault` |
| Dexter | `vault_personal` | Configured on Dexter's machine |

## Installation

```bash
cd vault-embedder
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `config.yaml`:

```yaml
vault_path: /path/to/obsidian/vault
collection_name: vault_work
qdrant_url: http://localhost:6333

# Optional
exclude_patterns:
  - ".*"           # hidden files
  - "_*"           # underscore prefixed
  - "templates/*"  # template folder
  
include_extensions:
  - .md
  
chunk_size: 1000      # max chars per chunk
chunk_overlap: 200    # overlap between chunks
```

## Usage

```bash
# Index entire vault
python -m vault_embedder index

# Index specific file
python -m vault_embedder index path/to/file.md

# Search
python -m vault_embedder search "your query" --limit 10

# Status
python -m vault_embedder status
```

## Integration with OpenClaw

This tool is designed to be called from OpenClaw skills or directly via CLI. The search results return:

```json
{
  "results": [
    {
      "path": "Projects/foo.md",
      "chunk": "Relevant text...",
      "score": 0.85,
      "metadata": {
        "heading": "## Section Name",
        "line_start": 42,
        "line_end": 67
      }
    }
  ]
}
```

## Model Details

- **Model**: `taylorai/bge-micro-v2`
- **Dimensions**: 384
- **Size**: ~24MB
- **Source**: Same model used by Smart Connections for Obsidian
- **License**: Apache 2.0

## Development

This is part of the `openclaw-unified-plugins` repo. Steve handles embedder core, Dexter handles Qdrant setup + search wrapper.
