import fs from 'fs';
import path from 'path';
import hnswlib from 'hnswlib-node';

/**
 * HNSW Vector Store implementation for Xenova embeddings
 */
export class HNSWVectorStore {
  constructor(options = {}) {
    this.generateEmbedding = options.generateEmbedding;
    this.index = null;
    this.documents = [];
    this.dimension = 384; // Default dimension for all-MiniLM-L6-v2
    this.maxElements = 10000; // Default size, will be increased as needed
    this.initialized = false;
    this.indexBuilt = false; // Track if the index has docs added
  }

  /**
   * Initialize the HNSW index
   */
  async initialize() {
    console.log("Initializing HNSW index...");
    if (this.initialized) {
      console.log("HNSW index already initialized");
      return true;
    }
    
    try {
      // In hnswlib-node, the index is automatically built as points are added
      // There's no explicit buildIndex method
      this.index = new hnswlib.HierarchicalNSW('cosine', this.dimension);
      this.index.initIndex(this.maxElements, 16, 200);
      this.initialized = true;
      console.log('HNSW index initialized successfully');
      return true;
    } catch (error) {
      console.error(`Failed to initialize HNSW index: ${error.message}`);
      this.initialized = false;
      return false;
    }
  }

  /**
   * Add documents to the vector store
   * @param {Array} documents Array of documents with pageContent and metadata
   */
  async addDocuments(documents) {
    if (!this.initialized && !(await this.ensureInitialized())) {
      console.error("Failed to initialize HNSW index before adding documents");
      return;
    }

    if (!documents || documents.length === 0) return;

    // Expand index if needed
    if (this.documents.length + documents.length > this.maxElements) {
      const newSize = Math.max(this.maxElements * 2, this.documents.length + documents.length);
      this.index.resizeIndex(newSize);
      this.maxElements = newSize;
      console.log(`Resized index to ${newSize} elements`);
    }
    
    let docsProcessed = 0;
    for (const doc of documents) {
      try {
        // Skip documents without content
        if (!doc.pageContent) continue;
        
        // Generate embedding for the document
        const embedding = await this.generateEmbedding(doc.pageContent);
        if (!embedding) {
          console.warn("⚠️ Failed to generate embedding for document:", doc);
          continue;
        }
        
        // Ensure embedding is a numeric array
        const embeddingArray = this.ensureEmbeddingIsArray(embedding);
        if (!embeddingArray) {
          console.warn("⚠️ Could not convert embedding to array for document:", doc);
          continue;
        }

        // Add to index - this automatically builds the index in hnswlib-node
        const docIndex = this.documents.length;
        this.index.addPoint(embeddingArray, docIndex);
        this.documents.push(doc);
        this.indexBuilt = true; // Mark that we have added points

        docsProcessed++;
        if (docsProcessed % 1000 === 0) {
          console.log(`✅ Processed ${docsProcessed} documents`);
        }  
      } catch (error) {
        console.error(`Error adding document to vector store: ${error.message}`);
      }
    }
    
    console.log(`Added ${docsProcessed} documents to vector store`);
  }

  /**
   * Ensure the embedding is a plain numeric array
   * @param {any} embedding The embedding to check and convert
   * @returns {Float32Array|Array|null} The embedding as an array or null if conversion failed
   */
  ensureEmbeddingIsArray(embedding) {
    if (!embedding) return null;
    
    // If it's already an array, return it
    if (Array.isArray(embedding)) return embedding;
    
    // If it's a Float32Array, return it
    if (embedding instanceof Float32Array) return embedding;
    
    try {
      // If it's a tensor-like object with data or values property
      if (embedding.data) return Array.from(embedding.data);
      if (embedding.values) return Array.from(embedding.values);
      
      // If it has a toArray method
      if (typeof embedding.toArray === 'function') return embedding.toArray();
      
      // If it has a flatten method
      if (typeof embedding.flatten === 'function') return embedding.flatten();
      
      // Last resort: try to convert to array
      return Array.from(embedding);
    } catch (error) {
      console.error(`Failed to convert embedding to array: ${error.message}`);
      return null;
    }
  }

  /**
   * Perform similarity search
   * @param {string} query Query text
   * @param {number} k Number of results to return
   * @returns {Array} Array of documents with similarity score
   */
  async similaritySearch(query, k = 5) {
    if (!this.initialized || this.documents.length === 0 || !this.indexBuilt) {
      console.warn('Vector store not initialized, empty, or no index built');
      return [];
    }

    try {
      // Generate embedding for the query
      const queryEmbedding = await this.generateEmbedding(query);
      if (!queryEmbedding) {
        console.error('Failed to generate embedding for query');
        return [];
      }

      // Ensure embedding is an array
      const queryEmbeddingArray = this.ensureEmbeddingIsArray(queryEmbedding);
      if (!queryEmbeddingArray) {
        console.error('Failed to convert query embedding to array');
        return [];
      }

      // Limit k to the number of documents
      const effectiveK = Math.min(k, this.documents.length);
      if (effectiveK === 0) return [];

      // Search the index
      const result = this.index.searchKnn(queryEmbeddingArray, effectiveK);
      
      // Map results to documents
      return result.neighbors.map((docIndex, i) => {
        return {
          ...this.documents[docIndex],
          score: result.distances[i]
        };
      });
    } catch (error) {
      console.error(`Search error: ${error.message}`);
      return [];
    }
  }

  /**
   * Save the vector store to disk
   * @param {string} directory Directory to save to
   */
  async save(filepath) {
    try {
      // Create directory if it doesn't exist
      const dir = path.dirname(filepath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      // Save the index
      const indexPath = `${filepath}.index`;
      this.index.writeIndex(indexPath);
      
      // Save documents and metadata
      const metadataPath = `${filepath}.json`;
      fs.writeFileSync(metadataPath, JSON.stringify({
        documents: this.documents,
        dimension: this.dimension,
        maxElements: this.maxElements,
        indexBuilt: this.indexBuilt
      }));
      
      console.log(`Vector store saved to ${filepath}`);
      return true;
    } catch (error) {
      console.error(`Error saving vector store: ${error.message}`);
      return false;
    }
  }

  /**
   * Load the vector store from disk
   * @param {string} directory Directory to load from
   */
  async load(filepath) {
    try {
      // Check if files exist
      const indexPath = `${filepath}.index`;
      const metadataPath = `${filepath}.json`;
      
      if (!fs.existsSync(indexPath) || !fs.existsSync(metadataPath)) {
        console.error('Vector store files not found');
        return false;
      }
      
      // Load metadata
      const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
      this.documents = metadata.documents;
      this.dimension = metadata.dimension;
      this.maxElements = metadata.maxElements;
      this.indexBuilt = metadata.indexBuilt || this.documents.length > 0;
      
      // Initialize index
      this.index = new hnswlib.HierarchicalNSW('cosine', this.dimension);
      
      // Load index
      this.index.readIndex(indexPath, 100);
      
      this.initialized = true;
      
      console.log(`Loaded vector store with ${this.documents.length} documents`);
      return true;
    } catch (error) {
      console.error(`Error loading vector store: ${error.message}`);
      return false;
    }
  }
}