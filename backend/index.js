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
  } catch (error) {
    console.error(`Database error: ${error.message}`);
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

// Setup vector store
/**
 * Setup vector store with proper initialization
 * @returns {Promise<boolean>} Success state of vector store setup
 */
/**
 * Setup vector store with proper initialization, checkpoints, and chunk processing
 * @returns {Promise<boolean>} Success state of vector store setup
 */
async function setupVectorStore() {
  try {
    // Ensure embedding model is initialized
    if (!embedder) {
      const modelInitialized = await initEmbeddingModel();
      if (!modelInitialized) {
        console.error("Cannot proceed with vector store setup - embedding model failed to initialize");
        return false;
      }
    }

    const vectorStoreDir = path.join(__dirname, 'vectorstore');
    if (!fs.existsSync(vectorStoreDir)) {
      fs.mkdirSync(vectorStoreDir, { recursive: true });
    }

    const vectorStorePath = path.join(vectorStoreDir, 'vectorstore.hnsw');
    const checkpointPath = path.join(vectorStoreDir, 'checkpoint.json');

    // Create a new instance of your vector store with the embedding function
    vectorStore = new HNSWVectorStore({
      generateEmbedding: generateEmbedding
    });

    // Initialize the vector store explicitly
    await vectorStore.initialize();
    
    console.log('Vector store instance created');

    // Load vector store from disk if available
    if (fs.existsSync(`${vectorStorePath}.index`) && fs.existsSync(`${vectorStorePath}.json`)) {
      console.log('Attempting to load vector store from disk...');
      const loaded = await vectorStore.load(vectorStorePath);
      if (loaded) {
        console.log(`Successfully loaded vector store with ${vectorStore.documents.length} documents`);
        return true;
      } else {
        console.warn('Failed to load vector store from disk, will build a new one');
      }
    } else {
      console.log('No existing vector store found, will build a new one');
    }

    console.log('Building new vector store...');
    
    // Get product count to monitor progress
    const productCount = await db.get('SELECT COUNT(*) as count FROM products');
    console.log(`Building vector store from ${productCount.count} products`);

    if (productCount.count === 0) {
      console.warn('No products in database, cannot build vector store');
      return false;
    }

    // Check for existing checkpoint
    let offset = 0;
    let totalProcessed = 0;
    
    if (fs.existsSync(checkpointPath)) {
      try {
        const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8'));
        offset = checkpoint.offset || 0;
        totalProcessed = checkpoint.processed || 0;
        console.log(`Resuming from checkpoint: processed ${totalProcessed} products, starting at offset ${offset}`);
      } catch (error) {
        console.warn(`Error reading checkpoint, starting from beginning: ${error.message}`);
      }
    }

    // Process in chunks with checkpoints
    const CHUNK_SIZE = 5000; // Process 5000 rows at a time
    const CHECKPOINT_INTERVAL = 1000; // Save checkpoint every 1000 documents
    const SAVE_INTERVAL = 10000; // Save the vector store every 10000 documents
    
    let lastCheckpoint = totalProcessed;
    let lastSave = totalProcessed;
    let batch = [];

    while (true) {
      // Fetch products in chunks
      const products = await db.all(
        `SELECT asin, title, brand, description, price, sales_rank, review_rating, review_count 
         FROM products ORDER BY asin LIMIT ? OFFSET ?`, 
        [CHUNK_SIZE, offset]
      );
      
      if (products.length === 0) break;
      
      console.log(`Processing chunk of ${products.length} products starting at offset ${offset}`);
      
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

        // Process in batches of EMBEDDING_BATCH_SIZE
        if (batch.length >= EMBEDDING_BATCH_SIZE) {
          await vectorStore.addDocuments(batch);
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

      // Process any remaining items in this chunk
      if (batch.length > 0) {
        await vectorStore.addDocuments(batch);
        totalProcessed += batch.length;
        batch = [];
      }
      
      // Update offset for next chunk
      offset += products.length;
      
      // Save checkpoint after each chunk
      fs.writeFileSync(checkpointPath, JSON.stringify({
        offset: offset,
        processed: totalProcessed,
        timestamp: new Date().toISOString()
      }));
      console.log(`Processed ${totalProcessed}/${productCount.count} products for vector store`);
      console.log(`Chunk complete. Checkpoint saved at offset ${offset}`);
      
      // Save vector store after each chunk to ensure persistence
      await vectorStore.save(vectorStorePath);
      console.log(`Vector store saved with ${vectorStore.documents.length} documents`);
    }

    // Save the final vector store
    console.log(`Vector store built with ${vectorStore.documents.length} documents, saving to disk...`);
    await vectorStore.save(vectorStorePath);
    console.log('Vector store saved to disk');
    
    // Clear the checkpoint file after successful completion
    if (fs.existsSync(checkpointPath)) {
      fs.unlinkSync(checkpointPath);
      console.log('Checkpoint file removed - process complete');
    }
    
    return true;
  } catch (error) {
    console.error(`Vector store setup error: ${error.message}`);
    console.error(`Full error:`, error);
    return false;
  }
}

// Server initialization
async function initServer() {
  try {
    console.log('Starting server initialization...');
    
    // Load the embedding model first
    const modelLoaded = await initEmbeddingModel();
    if (!modelLoaded) {
      console.error('Failed to load embedding model, cannot proceed');
      process.exit(1);
    }
    
    // Set up the database
    await setupDatabase();
    console.log('Database setup complete');
    
    // Analyze CSV files
    await analyzeCSVFiles();
    console.log('CSV analysis complete');
    
    // Import CSV files if they haven't been processed
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      if (metadata.recordCount > 0 && !metadata.processed) {
        console.log(`Importing data from ${filename}...`);
        await importCSVToDatabase(metadata.path, filename);
        metadata.processed = true;
      }
    }
    
    // IMPORTANT: Setup vector store before API routes
    console.log('Setting up vector store...');
    const vectorStoreReady = await setupVectorStore();
    if (!vectorStoreReady) {
      console.warn('Vector store setup incomplete - search functionality may be limited');
    } else {
      console.log('Vector store setup complete and ready for search');
    }
    
    // Setup API routes after vector store is ready
    console.log('Setting up API routes...');
    function setupAPIRoutes() {
      // Example search endpoint
      app.get('/search', async (req, res) => {
        const query = req.query.q;
        if (!query || typeof query !== 'string') {
          return res.status(400).json({ error: 'Missing search query' });
        }
    
        try {
          const results = await vectorStore.similaritySearch(query, 10); // Top 10 results
          res.json({ results });
        } catch (error) {
          console.error('Search error:', error);
          res.status(500).json({ error: 'Search failed' });
        }
      });
    
      // Optional: Health check endpoint
      app.get('/health', (req, res) => {
        res.json({
          status: 'healthy',
          vectorStore: vectorStore?.initialized ? 'ready' : 'not initialized',
          documents: vectorStore?.documents?.length || 0
        });
      });
    }
    
    setupAPIRoutes();
    
    // Start server
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`Server is ready and running on port ${PORT}`);
      console.log(`CSV directory: ${CSV_DIR}`);
      console.log(`Database: ${DB_PATH}`);
      console.log(`Vector store status: ${vectorStore && vectorStore.initialized ? 'Initialized' : 'Not initialized'}`);
      console.log(`Vector store documents: ${vectorStore ? vectorStore.documents.length : 0}`);
    });
  } catch (error) {
    console.error(`Server initialization error: ${error.message}`);
    console.error(error);
    process.exit(1);
  }
}


// Start the server
initServer();