# remove_large_files.py
def filter_file(path, blob, commit):
    # Exclude vectorstore files
    if path.startswith(b"backend/vectorstore/vectorstore.hnsw") or \
       path == b"backend/vectorstore/checkpoint.json":
        return None  # Remove this file
    return blob