import fs from 'fs';
import path from 'path';
import hnswlib from 'hnswlib-node';

/**
 * HNSW Vector Store implementation for Xenova embeddings
 * With complete index management and error handling
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
    this.spaceName = 'cosine'; // Distance metric
    this.efConstruction = 200; // Index quality parameter
    this.M = 16; // Index quality parameter
  }

  /**
   * Initialize the HNSW index from scratch
   */
  async initialize() {
    console.log("Initializing HNSW index...");
    
    try {
      // Always recreate the index when initializing
      this.index = new hnswlib.HierarchicalNSW(this.spaceName, this.dimension);
      this.index.initIndex(this.maxElements, this.M, this.efConstruction);
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
    if (!this.initialized || !this.index) {
      console.log("Index not initialized, initializing now...");
      if (!await this.initialize()) {
        console.error("Failed to initialize index, cannot add documents");
        return false;
      }
    }

    if (!documents || documents.length === 0) return true;

    // Expand index if needed
    if (this.documents.length + documents.length > this.maxElements) {
      try {
        const newSize = Math.max(this.maxElements * 2, this.documents.length + documents.length);
        console.log(`Resizing index from ${this.maxElements} to ${newSize} elements`);
        this.index.resizeIndex(newSize);
        this.maxElements = newSize;
        console.log(`Index resized to ${newSize} elements`);
      } catch (error) {
        console.error(`Failed to resize index: ${error.message}`);
        return false;
      }
    }
    
    let docsProcessed = 0;
    let batchSuccessful = true;
    
    for (const doc of documents) {
      try {
        // Skip documents without content
        if (!doc.pageContent) continue;
        
        // Generate embedding for the document
        const embedding = await this.generateEmbedding(doc.pageContent);
        if (!embedding) {
          console.warn("⚠️ Failed to generate embedding for document");
          continue;
        }
        
        // Ensure embedding is a numeric array
        const embeddingArray = this.ensureEmbeddingIsArray(embedding);
        if (!embeddingArray) {
          console.warn("⚠️ Could not convert embedding to array for document");
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
        batchSuccessful = false;
      }
    }
    
    console.log(`Added ${docsProcessed} documents to vector store`);
    return batchSuccessful;
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
   * Check if the index is ready for search
   * @returns {boolean} True if ready for search
   */
  isSearchable() {
    return this.initialized && this.index && this.indexBuilt && this.documents.length > 0;
  }

  /**
   * Perform similarity search
   * @param {string} query Query text
   * @param {number} k Number of results to return
   * @returns {Array} Array of documents with similarity score
   */
  async similaritySearch(query, k = 5) {
    if (!this.isSearchable()) {
      console.warn('Vector store not ready for search - not initialized or empty');
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
   * @param {string} filepath Base filepath to save to
   */
  async save(filepath) {
    try {
      // Ensure directory exists
      const dir = path.dirname(filepath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      if (!this.initialized || !this.index) {
        console.warn("Cannot save - vector store not initialized");
        return false;
      }
      
      if (this.documents.length === 0) {
        console.warn("Cannot save - vector store has no documents");
        return false;
      }
      
      // Save the index
      const indexPath = `${filepath}.index`;
      this.index.writeIndex(indexPath);
      
      // Save documents and metadata
      const metadataPath = `${filepath}.json`;
      const metadata = {
        documents: this.documents,
        dimension: this.dimension,
        maxElements: this.maxElements,
        indexBuilt: this.indexBuilt,
        spaceName: this.spaceName,
        efConstruction: this.efConstruction,
        M: this.M
      };
      
      fs.writeFileSync(metadataPath, JSON.stringify(metadata));
      
      console.log(`Vector store saved to ${filepath} with ${this.documents.length} documents`);
      return true;
    } catch (error) {
      console.error(`Error saving vector store: ${error.message}`);
      return false;
    }
  }

  /**
   * Load the vector store from disk
   * @param {string} filepath Base filepath to load from
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
      
      console.log(`Loading index from ${indexPath} and metadata from ${metadataPath}`);
      
      // Load metadata first
      const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
      this.documents = metadata.documents || [];
      this.dimension = metadata.dimension || this.dimension;
      this.maxElements = metadata.maxElements || this.maxElements;
      this.indexBuilt = metadata.indexBuilt || this.documents.length > 0;
      this.spaceName = metadata.spaceName || this.spaceName;
      this.efConstruction = metadata.efConstruction || this.efConstruction;
      this.M = metadata.M || this.M;
      
      console.log(`Metadata loaded: dimension=${this.dimension}, maxElements=${this.maxElements}, documents=${this.documents.length}`);
      
      // Create a fresh index
      try {
        this.index = new hnswlib.HierarchicalNSW(this.spaceName, this.dimension);
        
        // Load the index from file - pass only the exact params needed
        // This is the critical fix: properly setting up the index before reading from disk
        console.log(`Reading index with maxElements=${this.maxElements}`);
        this.index.readIndex(indexPath, this.maxElements);
        
        // Set search parameters after loading
        this.index.setEf(Math.max(this.efConstruction, 50)); // Set search quality parameter
        
        this.initialized = true;
        console.log(`Loaded vector store with ${this.documents.length} documents successfully`);
        return true;
      } catch (error) {
        console.error(`Error reading index: ${error.message}`);
        // Reset the object state
        this.index = null;
        this.initialized = false;
        return false;
      }
    } catch (error) {
      console.error(`Error loading vector store: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Reset the vector store and force re-initialization
   */
  async reset() {
    this.index = null;
    this.documents = [];
    this.initialized = false;
    this.indexBuilt = false;
    
    return await this.initialize();
  }
}