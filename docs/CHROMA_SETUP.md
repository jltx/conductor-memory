# Chroma Setup Instructions

## Option 1: Local Persistence Mode (Recommended for Development)
No additional setup required! Chroma will automatically create a local database in `./data/chroma/`

```bash
# Just install dependencies
pip install -r requirements.txt

# Run the system - Chroma will create local database automatically
python your_main_script.py
```

## Option 2: Chroma Server Mode (For Production/Performance)
If you want a dedicated Chroma server:

```bash
# Install Docker, then run Chroma server
docker run -p 8000:8000 chromadb/chroma

# Set environment variables
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
```

## Environment Variables (Optional)

```bash
# Chroma Configuration
export CHROMA_HOST=localhost          # For server mode
export CHROMA_PORT=8000              # For server mode
export CHROMA_PERSIST_DIR=./data/chroma  # Local persistence directory
export CHROMA_COLLECTION=memory_chunks   # Collection name

# Embedding Configuration
export EMBEDDING_MODEL=all-MiniLM-L6-v2  # Model to use
export EMBEDDING_DEVICE=cuda           # Use GPU (for RTX 4090)
export EMBEDDING_CACHE_DIR=./cache     # Model cache directory
```

## Verification

Test that Chroma is working:

```bash
python migrate_embeddings.py
```

This will create sample data and verify vector search functionality.