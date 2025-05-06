import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';
import { pipeline } from "@xenova/transformers";
import Papa from 'papaparse';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import { HNSWVectorStore } from './vectorstore/hnswStore.js'; // You'll need to implement this
import { env } from '@xenova/transformers';

// Set debug mode
env.debug = true;

// File path setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
dotenv.config();

// Configuration
const CSV_DIR = path.join(__dirname, '..', 'csv');
const VECTOR_STORE_PATH = path.join(__dirname, 'vectorstore');
const DB_PATH = path.join(__dirname, 'database', 'product_data.db');
const DB_DIR = path.dirname(DB_PATH);
const BATCH_SIZE = 100;
const EMBEDDING_BATCH_SIZE = 10; // 🔁 Increased for better throughput
const ESSENTIAL_FIELDS = [
  'ASIN', 'asin', 'Title', 'title', 'Brand', 'brand',
  'Description & Features: Description', 'Description & Features: Short Description',
  'Buy Box 🚚: Current', 'Sales Rank: Current',
  'Reviews: Rating', 'Reviews: Review Count'
];

// Express setup
const app = express();
app.use(cors());
app.use(express.json());

// State management
let csvMetadata = {};
let vectorStore = null;
let db = null;
let embedder = null;
let isVectorStoreInitializing = false;

// ✅ Set up basic routes first - before any complex initialization
function setupBasicRoutes() {
  // Health check endpoint that doesn't depend on the vector store
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'healthy',
      version: '1.0.0',
      services: {
        database: db ? 'connected' : 'not connected',
        vectorStore: vectorStore ? (vectorStore.initialized ? 'ready' : 'initializing') : 'not started',
        vectorStoreInitializing: isVectorStoreInitializing,
        model: embedder ? 'loaded' : 'not loaded'
      }
    });
  });

  // Vector search stub endpoint - returns a meaningful error when not ready
  app.post('/api/vectorsearch', async (req, res) => {
    const query = req.body.query || req.body.question;
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'Missing search query' });
    }

    // If vector store isn't ready, return a clear initialization status
    if (!vectorStore || !vectorStore.initialized) {
      return res.status(503).json({ 
        error: 'Vector store is still initializing',
        status: 'initializing',
        retryAfter: 10, // Suggest retry after 10 seconds
        query: query
      });
    }

    // Actual search logic (will only execute if vector store is ready)
    try {
      const results = await vectorStore.similaritySearch(query, 10); // Top 10 results
      
      res.json({ 
        results,
        query: query,
        status: 'success',
        count: results.length
      });
    } catch (error) {
      console.error('Search error:', error);
      res.status(500).json({ 
        error: 'Search failed', 
        message: error.message,
        status: 'error'
      });
    }
  });
}

// Initialize embedder model
async function initEmbeddingModel() {
  console.log('Starting model download...');
  try {
    // Load a local embedding model
    embedder = await pipeline("feature-extraction", "Xenova/gte-small");
    console.log("Embedding model loaded successfully", embedder);
    return true;
  } catch (error) {
    console.error(`Failed to load embedding model: ${error.message}`);
    return false;
  }
}

// Function to generate embeddings
async function generateEmbedding(text) {
  try {
    if (!text || typeof text !== 'string') {
      console.warn('Invalid input for embedding generation:', text);
      return null;
    }
    
    // Ensure we have meaningful text to embed (at least 3 chars)
    const trimmedText = text.trim();
    if (trimmedText.length < 3) {
      console.warn('Text too short for embedding generation');
      return null;
    }
    
    console.log(`Generating embedding for text (${trimmedText.length} chars)...`);
    
    const output = await embedder(trimmedText, {
      pooling: "mean",
      normalize: true,
    });
    
    // Debug log to understand the structure
    console.log("Embedding output type:", typeof output);
    console.log("Embedding output structure:", 
      output ? 
      Object.keys(output).map(key => `${key}: ${typeof output[key]}`) : 
      'null');
    
    // Get the embedding data array
    let embeddingArray = null;
    
    if (output && output.data) {
      // For some models, output.data is already an array
      if (Array.isArray(output.data)) {
        embeddingArray = output.data;
      } 
      // For typed arrays
      else if (output.data.buffer instanceof ArrayBuffer) {
        embeddingArray = Array.from(output.data);
      }
      // For tensor-like objects
      else if (typeof output.data === 'object') {
        embeddingArray = Array.from(Object.values(output.data));
      }
    }
    
    // For direct Float32Array result
    else if (output && output.buffer instanceof ArrayBuffer) {
      embeddingArray = Array.from(output);
    }
    // For direct array result
    else if (Array.isArray(output)) {
      embeddingArray = output;
    }
    // Last resort - examine the object for embeddings
    else if (output) {
      // Look for any array property that might contain embeddings
      for (const key of Object.keys(output)) {
        const value = output[key];
        if (Array.isArray(value) && value.length > 0 && 
            typeof value[0] === 'number') {
          embeddingArray = value;
          break;
        }
        
        // Check second level properties
        if (value && typeof value === 'object') {
          for (const subKey of Object.keys(value)) {
            const subValue = value[subKey];
            if (Array.isArray(subValue) && subValue.length > 0 && 
                typeof subValue[0] === 'number') {
              embeddingArray = subValue;
              break;
            }
          }
          if (embeddingArray) break;
        }
      }
    }
    
    // Validate the embedding array
    if (!embeddingArray || !Array.isArray(embeddingArray)) {
      console.error("Failed to extract embedding array from model output");
      console.error("Output dump:", JSON.stringify(output).slice(0, 200) + "...");
      return null;
    }
    
    // Check if we have numerical values
    if (embeddingArray.length === 0 || typeof embeddingArray[0] !== 'number') {
      console.error("Embedding array doesn't contain numerical values");
      return null;
    }
    
    console.log(`Successfully generated embedding vector of length ${embeddingArray.length}`);
    return embeddingArray;
    
  } catch (error) {
    console.error(`Embedding error: ${error.message}`);
    console.error(`Error stack:`, error.stack);
    return null;
  }
}
// Helper function to normalize row data
function normalizeRow(row) {
  const normalizedRow = {};
  for (const [key, value] of Object.entries(row)) {
    if (!value) continue;
    const trimmedKey = key.trim();
    const valueStr = value.toString().trim();
    
    if (ESSENTIAL_FIELDS.some(field => field === trimmedKey) || 
        (valueStr.length < 500 && trimmedKey.includes('Feature'))) {
      normalizedRow[trimmedKey] = valueStr;
    }
  }
  return normalizedRow;
}

// Setup database
async function setupDatabase() {
  if (!fs.existsSync(DB_DIR)) {
    fs.mkdirSync(DB_DIR, { recursive: true });
  }

  try {
    db = await open({
      filename: DB_PATH,
      driver: sqlite3.Database
    });

    await db.exec(`
      CREATE TABLE IF NOT EXISTS products (
        asin TEXT PRIMARY KEY,
        title TEXT,
        brand TEXT,
        description TEXT,
        price REAL,
        sales_rank INTEGER,
        review_rating REAL,
        review_count INTEGER,
        raw_data TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_products_title ON products(title);
    `);
    
    return true;
  } catch (error) {
    console.error(`Database error: ${error.message}`);
    return false;
  }
}

// Helper function to insert a batch of products
async function insertBatch(batch) {
  const stmt = await db.prepare(`
    INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  for (const item of batch) {
    await stmt.run([
      item.asin, item.title, item.brand, item.description,
      item.price, item.sales_rank, item.review_rating,
      item.review_count, item.raw_data
    ]);
  }

  await stmt.finalize();
}

// Analyze CSV files
async function analyzeCSVFiles() {
  csvMetadata = {};
  if (!fs.existsSync(CSV_DIR)) return;

  const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
  if (files.length === 0) return;

  const promises = files.map(file => new Promise((resolveFile) => {
    const filePath = path.join(CSV_DIR, file);
    csvMetadata[file] = {
      path: filePath,
      recordCount: 0,
      columns: [],
      sampleRows: [],
      processed: false
    };

    const readStream = fs.createReadStream(filePath, { encoding: 'utf8' });
    let headerProcessed = false;
    let rowCount = 0;

    Papa.parse(readStream, {
      header: true,
      skipEmptyLines: true,
      chunk: function(results) {
        if (!headerProcessed) {
          csvMetadata[file].columns = Object.keys(results.data[0] || {}).map(key => key.trim());
          headerProcessed = true;
        }
        rowCount += results.data.length;
        if (csvMetadata[file].sampleRows.length < 3) {
          csvMetadata[file].sampleRows.push(...results.data.slice(0, 3).map(normalizeRow));
        }
      },
      complete: function() {
        csvMetadata[file].recordCount = rowCount;
        csvMetadata[file].processed = false;
        resolveFile();
      },
      error: function(err) {
        console.error(`Error parsing ${file}: ${err.message}`);
        resolveFile();
      }
    });
  }));

  await Promise.all(promises);
}

// Import CSV to database
async function importCSVToDatabase(filePath, filename) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(filePath)) return resolve();

    const readStream = fs.createReadStream(filePath, { encoding: 'utf8' });
    let rowsProcessed = 0;
    let batch = [];

    // Start a single transaction
    db.run('BEGIN TRANSACTION')
      .catch(err => {
        console.error(`Failed to start transaction: ${err.message}`);
        reject(err);
      });

    Papa.parse(readStream, {
      header: true,
      skipEmptyLines: true,
      chunk: async function(results) {
        readStream.pause(); // Pause stream to avoid overwhelming

        try {
          for (const row of results.data) {
            const asin = row['ASIN'] || row['asin'];
            if (!asin) continue;

            const product = {
              asin,
              title: row['Title'] || row['title'],
              brand: row['Brand'] || row['brand'],
              description: row['Description & Features: Description'] || row['Description & Features: Short Description'],
              price: parseFloat(row['Buy Box 🚚: Current']),
              sales_rank: parseInt(row['Sales Rank: Current']),
              review_rating: parseFloat(row['Reviews: Rating']),
              review_count: parseInt(row['Reviews: Review Count']),
              raw_data: JSON.stringify(row)
            };

            batch.push(product);

            if (batch.length >= BATCH_SIZE) {
              // Insert batch and clear
              await insertBatch(batch);
              rowsProcessed += batch.length;
              batch = [];
            }
          }

          readStream.resume(); // Resume stream
        } catch (error) {
          console.error(`Error processing chunk: ${error.message}`);
          await db.run('ROLLBACK'); // Rollback on error
          reject(error);
        }
      },
      complete: async function() {
        try {
          // Insert any remaining rows in the final batch
          if (batch.length > 0) {
            await insertBatch(batch);
            rowsProcessed += batch.length;
            batch = [];
          }

          // Commit the transaction
          await db.run('COMMIT');
          console.log(`Imported ${rowsProcessed} rows from ${filename}`);
          resolve();
        } catch (error) {
          console.error(`Transaction commit failed: ${error.message}`);
          await db.run('ROLLBACK');
          reject(error);
        }
      },
      error: function(err) {
        console.error(`CSV parsing error: ${err.message}`);
        db.run('ROLLBACK')
          .catch(rollbackErr => console.error(`Rollback failed: ${rollbackErr.message}`));
        reject(err);
      }
    });
  });
}

/**
 * Setup vector store with proper initialization and robust error handling
 * @returns {Promise<boolean>} Success state of vector store setup
 */
async function setupVectorStore() {
  isVectorStoreInitializing = true;
  
  try {
    // Ensure embedding model is initialized
    if (!embedder) {
      const modelInitialized = await initEmbeddingModel();
      if (!modelInitialized) {
        console.error("Cannot proceed with vector store setup - embedding model failed to initialize");
        isVectorStoreInitializing = false;
        return false;
      }
    }

    const vectorStoreDir = path.join(__dirname, 'vectorstore');
    if (!fs.existsSync(vectorStoreDir)) {
      fs.mkdirSync(vectorStoreDir, { recursive: true });
    }

    const vectorStorePath = path.join(vectorStoreDir, 'vectorstore.hnsw');
    const checkpointPath = path.join(vectorStoreDir, 'checkpoint.json');

    vectorStore = new HNSWVectorStore({
      generateEmbedding: generateEmbedding
    });

    // Check for existing vector store files
    const indexFile = `${vectorStorePath}.index`;
    const jsonFile = `${vectorStorePath}.json`;

    if (fs.existsSync(indexFile) && fs.existsSync(jsonFile)) {
      console.log('Found existing vector store files. Attempting to load...');
      try {
        const loadedSuccessfully = await vectorStore.load(vectorStorePath);

        if (loadedSuccessfully) {
          console.log(`Successfully loaded vector store with ${vectorStore.documents.length} documents`);

          // Set search parameters for better results
          if (vectorStore.index) {
            try {
              vectorStore.index.setEf(100);
              console.log("Search parameters configured");
            } catch (efErr) {
              console.warn(`Could not set search parameters: ${efErr.message}`);
            }
          }

          // ✅ Don't check or use checkpoint logic if vector store already exists
          isVectorStoreInitializing = false;
          return true;
        }
      } catch (loadError) {
        console.error(`Error loading vector store: ${loadError.message}`);
      }
    }

    // If no existing vector store, proceed with building
    console.log('No existing vector store found or failed to load. Starting build process...');
    await vectorStore.reset(); // Reset to ensure clean state
    const buildResult = await buildVectorStore(vectorStorePath, checkpointPath);
    isVectorStoreInitializing = false;
    return buildResult;
  } catch (error) {
    console.error(`Vector store setup critical error: ${error.message}`);
    console.error(`Full error:`, error);
    isVectorStoreInitializing = false;
    return false;
  }
}

/**
 * Build the vector store from the database
 * @param {string} vectorStorePath Path to save the vector store
 * @param {string} checkpointPath Path for the checkpoint file
 * @returns {Promise<boolean>} Success state
 */
async function buildVectorStore(vectorStorePath, checkpointPath) {
  console.log('Building new vector store...');
  
  // Initialize vector store if needed
  if (!vectorStore.initialized) {
    console.log("Vector store not initialized, initializing now...");
    const initialized = await vectorStore.initialize();
    if (!initialized) {
      console.error("Failed to initialize vector store, cannot continue");
      return false;
    }
  }
  
  // Get product count to monitor progress
  const productCount = await db.get('SELECT COUNT(*) as count FROM products');
  console.log(`Building vector store from ${productCount.count} products`);

  if (productCount.count === 0) {
    console.warn('No products in database, cannot build vector store');
    return false;
  }

  // Load checkpoint if exists
  let offset = 0;
  let totalProcessed = 0;
  const backupCheckpointPath = `${checkpointPath}.backup`;
  
  if (fs.existsSync(checkpointPath)) {
    try {
      const checkpointData = fs.readFileSync(checkpointPath, 'utf-8');
      // Make a backup of the checkpoint file
      fs.writeFileSync(backupCheckpointPath, checkpointData);
      
      const checkpoint = JSON.parse(checkpointData);
      offset = checkpoint.offset || 0;
      totalProcessed = checkpoint.processed || 0;
      console.log(`Resuming from checkpoint: processed ${totalProcessed} products, starting at offset ${offset}`);
    } catch (error) {
      console.warn(`Error reading checkpoint: ${error.message}`);
      
      // Try to read from backup if available
      if (fs.existsSync(backupCheckpointPath)) {
        try {
          const backupCheckpoint = JSON.parse(fs.readFileSync(backupCheckpointPath, 'utf-8'));
          offset = backupCheckpoint.offset || 0;
          totalProcessed = backupCheckpoint.processed || 0;
          console.log(`Recovered from backup checkpoint: processed ${totalProcessed}, offset ${offset}`);
        } catch (backupError) {
          console.warn(`Backup checkpoint also corrupted: ${backupError.message}`);
          offset = 0;
          totalProcessed = 0;
        }
      } else {
        // Reset if no backup
        offset = 0;
        totalProcessed = 0;
      }
    }
  }

  // Processing configuration
  const CHUNK_SIZE = 5000; // Process rows in chunks
  const CHECKPOINT_INTERVAL = 1000; // Save checkpoint every N documents
  const SAVE_INTERVAL = 10000; // Save vector store every N documents
  const MAX_RETRIES = 3; // Maximum retries per chunk
  
  let lastCheckpoint = totalProcessed;
  let lastSave = totalProcessed;
  let batch = [];

  // Main processing loop
  let continueProcessing = true;
  let currentRetry = 0;
  
  while (continueProcessing) {
    try {
      // Fetch products in chunks
      const products = await db.all(
        `SELECT asin, title, brand, description, price, sales_rank, review_rating, review_count 
         FROM products ORDER BY asin LIMIT ? OFFSET ?`, 
        [CHUNK_SIZE, offset]
      );
      
      if (products.length === 0) {
        console.log("No more products to process, finishing up");
        break;
      }
      
      console.log(`Processing chunk of ${products.length} products starting at offset ${offset}`);
      
      // Process each product in the chunk
      for (const product of products) {
        const textContent = `
          Title: ${product.title || ''}\n
          Brand: ${product.brand || ''}\n
          Description: ${product.description || ''}\n
          Sales Rank: ${product.sales_rank || ''}\n
          Reviews: ${product.review_rating || ''}★ (${product.review_count || ''} reviews)\n
        `;

        // Truncate long content for performance
        const maxChars = 300;
        const trimmedContent = textContent.length > maxChars 
          ? textContent.slice(0, maxChars) + '...' 
          : textContent;

        batch.push({
          pageContent: trimmedContent,
          metadata: {
            asin: product.asin,
            title: product.title,
            source: 'database'
          }
        });

        // Process in batches
        if (batch.length >= EMBEDDING_BATCH_SIZE) {
          const success = await vectorStore.addDocuments(batch);
          if (!success) {
            console.warn("Batch processing had errors, but continuing");
          }
          
          totalProcessed += batch.length;
          batch = [];
          
          // Save checkpoint periodically
          if (totalProcessed - lastCheckpoint >= CHECKPOINT_INTERVAL) {
            fs.writeFileSync(checkpointPath, JSON.stringify({
              offset: offset + (totalProcessed - lastCheckpoint),
              processed: totalProcessed,
              timestamp: new Date().toISOString()
            }));
            lastCheckpoint = totalProcessed;
            console.log(`Checkpoint saved: ${totalProcessed}/${productCount.count} products processed`);
          }
          
          // Save vector store periodically
          if (totalProcessed - lastSave >= SAVE_INTERVAL) {
            console.log(`Saving vector store at ${totalProcessed} documents...`);
            await vectorStore.save(vectorStorePath);
            lastSave = totalProcessed;
            console.log(`Vector store saved with ${vectorStore.documents.length} documents`);
          }
        }
      }

      // Process any remaining items in batch
      if (batch.length > 0) {
        await vectorStore.addDocuments(batch);
        totalProcessed += batch.length;
        batch = [];
      }
      
      // Update offset for next chunk and reset retry counter
      offset += products.length;
      currentRetry = 0;
      
      // Save checkpoint after successful chunk
      fs.writeFileSync(checkpointPath, JSON.stringify({
        offset: offset,
        processed: totalProcessed,
        timestamp: new Date().toISOString()
      }));
      console.log(`Processed ${totalProcessed}/${productCount.count} products for vector store`);
      console.log(`Chunk complete. Checkpoint saved at offset ${offset}`);
      
      // Save vector store after each chunk
      await vectorStore.save(vectorStorePath);
      console.log(`Vector store saved with ${vectorStore.documents.length} documents`);
      
    } catch (error) {
      // Error handling for chunk processing
      console.error(`Error processing chunk: ${error.message}`);
      console.error(`Stack: ${error.stack}`);
      
      // Implement retry logic
      currentRetry++;
      if (currentRetry <= MAX_RETRIES) {
        console.log(`Retry attempt ${currentRetry}/${MAX_RETRIES}`);
        
        // Try to reinitialize the vector store
        console.log("Attempting to reset the vector store...");
        await vectorStore.reset();
        
        // Save current progress before retrying
        fs.writeFileSync(checkpointPath, JSON.stringify({
          offset: offset,
          processed: totalProcessed,
          timestamp: new Date().toISOString(),
          lastError: error.message,
          retry: currentRetry
        }));
        
        // Wait a moment before retrying
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Next iteration will retry this chunk
        continue;
      } else {
        console.error(`Max retries (${MAX_RETRIES}) exceeded for chunk at offset ${offset}`);
        
        // Save vector store with what we have so far
        if (vectorStore.documents.length > 0) {
          console.log(`Saving partial vector store with ${vectorStore.documents.length} documents`);
          await vectorStore.save(vectorStorePath);
        }
        
        // Update checkpoint with error info
        fs.writeFileSync(checkpointPath, JSON.stringify({
          offset: offset,
          processed: totalProcessed,
          timestamp: new Date().toISOString(),
          error: error.message,
          status: 'failed_with_max_retries'
        }));
        
        // Stop processing
        continueProcessing = false;
        break;
      }
    }
  }

  // Final save of the vector store
  if (vectorStore.documents.length > 0) {
    console.log(`Vector store built with ${vectorStore.documents.length} documents, saving final version...`);
    await vectorStore.save(vectorStorePath);
    console.log('Vector store saved to disk');
    
    // Only remove checkpoint if fully successful
    if (continueProcessing && fs.existsSync(checkpointPath)) {
      fs.unlinkSync(checkpointPath);
      console.log('Checkpoint file removed - process complete');
      if (fs.existsSync(backupCheckpointPath)) {
        fs.unlinkSync(backupCheckpointPath);
      }
    }
    
    return true;
  } else {
    console.warn('No documents were added to the vector store');
    return false;
  }
}

// ✅ Setup additional API routes after vectorstore is available
function setupAdvancedRoutes() {
  // Status endpoint with more detailed info
  app.get('/api/status', (req, res) => {
    res.json({
      status: 'online',
      vectorStore: vectorStore?.initialized ? 'ready' : 'not initialized',
      documents: vectorStore?.documents?.length || 0,
      vectorStoreInitializing: isVectorStoreInitializing
    });
  });

  // Example GET search endpoint (alternative for testing)
  app.get('/search', async (req, res) => {
    const query = req.query.q;
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'Missing search query' });
    }

    if (!vectorStore || !vectorStore.initialized) {
      return res.status(503).json({ 
        error: 'Vector store not initialized',
        status: 'service_unavailable'
      });
    }

    try {
      const results = await vectorStore.similaritySearch(query, 10); // Top 10 results
      res.json({ results });
    } catch (error) {
      console.error('Search error:', error);
      res.status(500).json({ error: 'Search failed' });
    }
  });
}

// ✅ Improved server initialization that won't hang if a step fails
async function initServer() {
  try {
    console.log('Starting server initialization...');
    
    // ✅ Setup basic routes immediately so the server can respond to requests
    console.log('Setting up basic API routes first...');
    setupBasicRoutes();
    
    // Start server early to handle requests during initialization
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT} (initialization in progress)`);
    });

    // 1. Database - Essential for basic functionality
    console.log('Setting up database...');
    const dbSuccess = await setupDatabase();
    if (!dbSuccess) {
      console.warn('Database setup had issues - some features may be limited');
    } else {
      console.log('Database setup complete');
    }
    
    // 2. Async initialize the embedding model
    console.log('Loading embedding model...');
    const modelLoaded = await initEmbeddingModel();
    if (!modelLoaded) {
      console.warn('Failed to load embedding model - search will not work');
    } else {
      console.log('Embedding model loaded successfully');
    }
    
    // 3. Analyze CSV files (non-critical)
    console.log('Analyzing CSV files...');
    await analyzeCSVFiles()
      .catch(err => console.warn(`CSV analysis error: ${err.message}`));
    console.log('CSV analysis complete');
    
    // 4. Import CSV files if they haven't been processed (non-critical)
    try {
      for (const [filename, metadata] of Object.entries(csvMetadata)) {
        if (metadata.recordCount > 0 && !metadata.processed) {
          console.log(`Importing data from ${filename}...`);
          await importCSVToDatabase(metadata.path, filename)
            .catch(err => console.warn(`CSV import error for ${filename}: ${err.message}`));
          metadata.processed = true;
        }
      }
    } catch (error) {
      console.warn(`CSV import process error: ${error.message}`);
    }
    
    // 5. Setup vector store in the background - allow server to function meanwhile
    console.log('Starting vector store setup in background...');
    setupVectorStore().then(vectorStoreReady => {
      if (!vectorStoreReady) {
        console.warn('Vector store setup incomplete - search functionality will be limited');
      } else {
        console.log('Vector store setup complete and ready for search');
      }
      
      // Setup advanced routes after vector store is ready
      setupAdvancedRoutes();
      
      console.log('Server initialization complete');
    }).catch(error => {
      console.error(`Vector store error: ${error.message}`);
      console.error(error);
    });
    
    // This log indicates the initialization process has started but may still be running in the background
    console.log('Server is responding to requests while initialization continues in the background');
    
  } catch (error) {
    console.error(`Server initialization error: ${error.message}`);
    console.error(error);
    
    // Even if initialization fails, keep the server running for diagnostics
    if (!app.listening) {
      const PORT = process.env.PORT || 5000;
      app.listen(PORT, () => {
        console.log(`Server running in degraded mode on port ${PORT} due to initialization error`);
      });
    }
  }
}

// Start the server
initServer();