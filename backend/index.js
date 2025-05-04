// Simplified Amazon Product Database and API Server
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import Papa from 'papaparse';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
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
const EMBEDDING_BATCH_SIZE = 20;
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

// Analyze CSV files
async function analyzeCSVFiles() {
  csvMetadata = {};
  if (!fs.existsSync(CSV_DIR)) return;
  
  const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
  if (files.length === 0) return;

  console.log(`Found ${files.length} CSV files`);
  
  for (const file of files) {
    const filePath = path.join(CSV_DIR, file);
    csvMetadata[file] = {
      path: filePath,
      recordCount: 0,
      columns: [],
      sampleRows: [],
      processed: false
    };

    try {
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
        }
      });
    } catch (err) {
      console.error(`Error reading ${file}: ${err.message}`);
    }
  }
}

// Import CSV to database
async function importCSVToDatabase(filePath, filename) {
  return new Promise((resolve) => {
    if (!fs.existsSync(filePath)) return resolve();
    
    const readStream = fs.createReadStream(filePath, { encoding: 'utf8' });
    let rowsProcessed = 0;
    let batch = [];
    
    Papa.parse(readStream, {
      header: true,
      skipEmptyLines: true,
      chunk: async function(results) {
        readStream.pause();
        
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
            await db.run('BEGIN TRANSACTION');
            for (const item of batch) {
              await db.run(`
                INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
              `, [
                item.asin, item.title, item.brand, item.description,
                item.price, item.sales_rank, item.review_rating,
                item.review_count, item.raw_data
              ]);
            }
            await db.run('COMMIT');
            
            rowsProcessed += batch.length;
            batch = [];
          }
        }
        
        readStream.resume();
      },
      complete: async function() {
        if (batch.length > 0) {
          await db.run('BEGIN TRANSACTION');
          for (const item of batch) {
            await db.run(`
              INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            `, [
              item.asin, item.title, item.brand, item.description,
              item.price, item.sales_rank, item.review_rating,
              item.review_count, item.raw_data
            ]);
          }
          await db.run('COMMIT');
          rowsProcessed += batch.length;
        }
        
        console.log(`Imported ${rowsProcessed} rows from ${filename}`);
        resolve();
      }
    });
  });
}

// Setup vector store
async function setupVectorStore() {
  if (!process.env.OPENAI_API_KEY) return;
  
  try {
    const embeddings = new OpenAIEmbeddings({ 
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "text-embedding-3-small",
      batchSize: EMBEDDING_BATCH_SIZE
    });
    
    vectorStore = new MemoryVectorStore(embeddings);
    
    // Process CSV files for vector store
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      if (metadata.recordCount === 0 || metadata.processed) continue;
      
      const filePath = metadata.path;
      const readStream = fs.createReadStream(filePath, 'utf8');
      let batch = [];
      
      Papa.parse(readStream, {
        header: true,
        skipEmptyLines: true,
        chunk: function(results) {
          results.data.forEach(row => {
            const textContent = `
              Title: ${row['Title'] || row['title']}\n
              Brand: ${row['Brand'] || row['brand']}\n
              Description: ${row['Description & Features: Description'] || row['Description & Features: Short Description']}\n
              Sales Rank: ${row['Sales Rank: Current']}\n
              Reviews: ${row['Reviews: Rating']}★ (${row['Reviews: Review Count']} reviews)\n
            `;
            
            batch.push({
              pageContent: textContent,
              metadata: {
                asin: row['ASIN'] || row['asin'],
                title: row['Title'] || row['title'],
                source: filename
              }
            });
            
            if (batch.length >= EMBEDDING_BATCH_SIZE) {
              vectorStore.addDocuments(batch);
              batch = [];
            }
          });
        },
        complete: function() {
          if (batch.length > 0) {
            vectorStore.addDocuments(batch);
          }
        }
      });
    }
  } catch (error) {
    console.error(`Vector store error: ${error.message}`);
  }
}

// API Routes
function setupAPIRoutes() {
  // Get CSV metadata
  app.get('/api/metadata', (req, res) => {
    res.json(csvMetadata);
  });
  
  // Database status endpoint
  app.get('/api/status', async (req, res) => {
    try {
      const productCount = await db.get('SELECT COUNT(*) as count FROM products');
      res.json({
        database: {
          path: DB_PATH,
          productCount: productCount.count
        },
        vectorStore: {
          initialized: !!vectorStore
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  // Product search endpoint
  app.get('/api/products', async (req, res) => {
    try {
      const { search } = req.query;
      let sql = 'SELECT asin, title, brand, price, sales_rank, review_rating FROM products';
      let params = [];
      
      if (search) {
        sql += ` WHERE title LIKE ? OR brand LIKE ? OR description LIKE ? OR asin LIKE ?`;
        const searchTerm = `%${search}%`;
        params = [searchTerm, searchTerm, searchTerm, searchTerm];
      }
      
      sql += ' ORDER BY sales_rank LIMIT 100';
      
      const results = await db.all(sql, params);
      res.json(results);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  // Product details endpoint
  app.get('/api/products/:asin', async (req, res) => {
    try {
      const result = await db.get('SELECT * FROM products WHERE asin = ?', [req.params.asin]);
      if (!result) return res.status(404).json({ error: 'Product not found' });
      
      // Parse raw JSON data
      if (result.raw_data) {
        result.details = JSON.parse(result.raw_data);
        delete result.raw_data;
      }
      
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
  
  // Vector search endpoint
  app.post('/api/vectorsearch', async (req, res) => {
    try {
      if (!vectorStore) return res.status(503).json({ error: 'Vector store not initialized' });
      const results = await vectorStore.similaritySearch(req.body.query, 5);
      res.json(results);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
}

// Server initialization
async function initServer() {
  try {
    await setupDatabase();
    await analyzeCSVFiles();
    
    // Import CSV files
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      if (metadata.recordCount > 0 && !metadata.processed) {
        await importCSVToDatabase(metadata.path, filename);
        metadata.processed = true;
      }
    }
    
    // Setup vector store
    await setupVectorStore();
    
    // Setup API routes
    setupAPIRoutes();
    
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      console.log(`CSV directory: ${CSV_DIR}`);
      console.log(`Database: ${DB_PATH}`);
    });
  } catch (error) {
    console.error(`Server error: ${error.message}`);
    process.exit(1);
  }
}

// Start the server
initServer();