// Integrated Amazon Product Database and API Server
// ES Modules format
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval";
import Papa from 'papaparse';
import pRetry from 'p-retry';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import _ from 'lodash';

// Get the directory name and filename

let __filename = fileURLToPath(import.meta.url);
let __dirname = dirname(__filename);

// Import HNSWLib conditionally
let HNSWLib;
try {
  const { HNSWLib: HNSW } = await import("@langchain/community/vectorstores/hnswlib");
  HNSWLib = HNSW;
} catch (error) {
  console.warn("Failed to load HNSWLib, falling back to MemoryVectorStore:", error.message);
  HNSWLib = null;
}

/**
 * Enhanced Logging Setup
 */
import { createLogger, format, transports } from 'winston';

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Create logger configuration
const logger = createLogger({
  level: 'debug',
  format: format.combine(
    format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    format.errors({ stack: true }),
    format.splat(),
    format.json()
  ),
  defaultMeta: { service: 'amazon-product-db' },
  transports: [
    // Write to console with colors
    new transports.Console({
      format: format.combine(
        format.colorize(),
        format.printf(info => `${info.timestamp} ${info.level}: ${info.message}${info.stack ? '\n' + info.stack : ''}`)
      )
    }),
    // Write all logs to log files
    new transports.File({ 
      filename: path.join(logsDir, 'error.log'), 
      level: 'error',
      maxsize: 10485760, // 10MB
      maxFiles: 5,
    }),
    new transports.File({ 
      filename: path.join(logsDir, 'combined.log'),
      maxsize: 10485760, // 10MB
      maxFiles: 5,
    })
  ]
});

// Log that the logger was initialized
logger.info('Winston logger initialized');

// Replace console.log, console.warn, console.error with logger
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

console.log = (...args) => {
  logger.info(args.join(' '));
  originalConsoleLog(...args);
};

console.warn = (...args) => {
  logger.warn(args.join(' '));
  originalConsoleWarn(...args);
};

console.error = (...args) => {
  logger.error(args.join(' '));
  originalConsoleError(...args);
};


dotenv.config();


// Configuration
const CSV_DIR = path.join(__dirname, '..', 'csv');
const VECTOR_STORE_PATH = path.join(__dirname, 'vectorstore');
const DB_PATH = path.join(__dirname, 'database', 'product_data.db');
const BATCH_SIZE = 100; // Smaller batch size for processing
const EMBEDDING_BATCH_SIZE = 25; // Number of rows to embed at once
const MAX_CONCURRENT_BATCHES = 2; // Control concurrent API calls
const USE_PERSISTENT_VECTOR_STORE = true; // Set to true to use HNSWLib instead of MemoryVectorStore

// Express setup
const app = express();
app.use(cors());
app.use(express.json());

// State management
let csvMetadata = {};
let vectorStore = null;
let isProcessingCSVs = false;
let db = null; // Database connection

// Helper function to sleep
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Helper function to normalize row data
function normalizeRow(row) {
  const normalizedRow = {};
  for (const [key, value] of Object.entries(row)) {
    const trimmedKey = key.trim();
    normalizedRow[trimmedKey] = value !== undefined && value !== null ? value.toString().trim() : null;
  }
  return normalizedRow;
}

/**
 * Database Setup and Initialization
 */
async function setupDatabase() {
  console.log('Starting database setup...');
  
  // Ensure the database directory exists
  const dbDir = path.dirname(DB_PATH);
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
    console.log(`Created database directory at ${dbDir}`);
  } else {
    console.log(`Database directory exists at ${dbDir}`);
  }

  // Check if database file already exists
  const dbExists = fs.existsSync(DB_PATH);
  console.log(`Database file ${dbExists ? 'exists' : 'does not exist'} at ${DB_PATH}`);

   // Open the database connection
   try {
    db = await open({
      filename: DB_PATH,
      driver: sqlite3.Database
    });
    console.log(`Successfully opened database connection to ${DB_PATH}`);
  } catch (error) {
    console.error(`Failed to open database connection: ${error.message}`);
    throw error;
  }

  // Create tables if they don't exist
  
  try {
    console.log('Creating database tables if they don\'t exist...');
    await db.exec(`
      CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_source TEXT,
        asin TEXT UNIQUE,
        title TEXT,
        brand TEXT,
        description TEXT,
        price REAL,
        list_price REAL,
        sales_rank INTEGER,
        review_rating REAL,
        review_count INTEGER,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    
      CREATE TABLE IF NOT EXISTS product_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        key TEXT,
        value TEXT,
        FOREIGN KEY (product_id) REFERENCES products(id)
      );
    
      CREATE INDEX IF NOT EXISTS idx_products_asin ON products(asin);
      CREATE INDEX IF NOT EXISTS idx_metadata_product_id ON product_metadata(product_id);
    `);

  // Check if tables were created
  const tables = await db.all("SELECT name FROM sqlite_master WHERE type='table'");
  console.log(`Database tables: ${tables.map(t => t.name).join(', ')}`);
  
  console.log("Database setup complete");
} catch (error) {
  console.error(`Failed to create database tables: ${error.message}`);
  throw error;
}
return db;
}

/**
 * CSV Analysis and Processing
 */
async function analyzeCSVFiles() {
  return new Promise(async (resolve) => {
    csvMetadata = {};
    
    // Check if CSV directory exists
    if (!fs.existsSync(CSV_DIR)) {
      console.warn(`CSV directory not found: ${CSV_DIR}`);
      return resolve({});
    } else {
      console.log(`Found CSV directory at ${CSV_DIR}`);
    }
    
    // Get all CSV files
    const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
    if (files.length === 0) {
      console.warn('No CSV files found in the directory');
      return resolve({});
    } else {
      console.log(`Found ${files.length} CSV files: ${files.join(', ')}`);
    }
    
    let filesProcessed = 0;
    
    for (const file of files) {
      const filePath = path.join(CSV_DIR, file);
      console.log(`Starting analysis of CSV file: ${file}`);
      
      // Initialize metadata structure
      csvMetadata[file] = {
        path: filePath,
        recordCount: 0,
        columns: [],
        sampleRows: [],
        processed: false
      };
      
      // Create a read stream for the file
      try {
        const readStream = fs.createReadStream(filePath, { encoding: 'utf8' });
        console.log(`Successfully created read stream for ${file}`);
        
        // Use Papa Parse in streaming mode
        let rowCount = 0;
        let headerProcessed = false;
        
        Papa.parse(readStream, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          delimitersToGuess: [',', '\t', '|', ';'],
          
          chunk: function(results) {
            if (results.errors && results.errors.length > 0) {
              console.warn(`Parsing warnings in ${file} chunk: ${JSON.stringify(results.errors)}`);
            }
            
            if (!headerProcessed && results.data.length > 0) {
              // Process headers
              const firstRow = results.data[0];
              csvMetadata[file].columns = Object.keys(firstRow).map(key => key.trim());
              console.log(`Found ${csvMetadata[file].columns.length} columns in ${file}: ${csvMetadata[file].columns.slice(0, 5).join(', ')}${csvMetadata[file].columns.length > 5 ? '...' : ''}`);
              headerProcessed = true;
            }
            
            // Count records and collect sample rows
            rowCount += results.data.length;
            console.log(`Processed chunk of ${results.data.length} rows from ${file}, total so far: ${rowCount}`);
            
            // Only keep first 5 rows as samples
            if (csvMetadata[file].sampleRows.length < 5) {
              const normalizedSamples = results.data
                .slice(0, 5 - csvMetadata[file].sampleRows.length)
                .map(normalizeRow);
              csvMetadata[file].sampleRows.push(...normalizedSamples);
              console.log(`Added ${normalizedSamples.length} sample rows from ${file}`);
            }
          },
          
          complete: function() {
            csvMetadata[file].recordCount = rowCount;
            filesProcessed++;
            console.log(`Analysis of ${file} complete: ${rowCount} records found`);
            
            if (filesProcessed === files.length) {
              console.log(`Completed analysis of all ${files.length} CSV files`);
              resolve(csvMetadata);
            }
          },
          
          error: function(error) {
            console.error(`Error analyzing ${file}: ${error.message}`);
            filesProcessed++;
            
            if (filesProcessed === files.length) {
              console.log(`Completed analysis of all ${files.length} CSV files with some errors`);
              resolve(csvMetadata);
            }
          }
        });
      } catch (err) {
        console.error(`Error reading file ${file}: ${err.message}`);
        filesProcessed++;
        
        if (filesProcessed === files.length) {
          console.log(`Completed analysis of all ${files.length} CSV files with some errors`);
          resolve(csvMetadata);
        }
      }
    }
  });
}

/**
 * Database Operations
 */
async function saveBatchToDatabase(batch, filename) {
  let inserted = 0;
  let updated = 0;
  let skipped = 0;
  let errors = 0;
  
  try {
    console.log(`Starting transaction to insert/update ${batch.length} products from ${filename}`);
    await db.run('BEGIN TRANSACTION');

    const stmt = await db.prepare(`
      INSERT OR REPLACE INTO products (
        id, file_source, asin, title, brand, description,
        price, list_price, sales_rank, review_rating, review_count
      ) VALUES (
        (SELECT id FROM products WHERE asin = ?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
      )
    `);

    const metaStmt = await db.prepare(`
      INSERT OR REPLACE INTO product_metadata (product_id, key, value)
      VALUES (?, ?, ?)
    `);

    // Check for duplicate ASINs within the batch
    const asinSet = new Set();
    const duplicatesInBatch = [];

    for (const row of batch) {
      const asin = row['ASIN'] || row['asin'] || null;
      
      if (!asin) {
        console.warn(`Skipping row in ${filename} - missing ASIN: ${JSON.stringify(Object.keys(row))}`);
        skipped++;
        continue;
      }
      
      // Check for duplicates within the batch
      if (asinSet.has(asin)) {
        duplicatesInBatch.push(asin);
        console.log(`Found duplicate ASIN ${asin} within current batch from ${filename}`);
        continue;
      }
      
      asinSet.add(asin);

      // Check if product already exists in database
      const existingProduct = await db.get('SELECT id FROM products WHERE asin = ?', [asin]);
      
      const title = row['Title'] || row['title'] || null;
      const brand = row['Brand'] || row['brand'] || null;
      const description = row['Description & Features: Description'] ||
                          row['Description & Features: Short Description'] || null;

      const price = parseFloat(row['Buy Box 🚚: Current']) || null;
      const listPrice = parseFloat(row['List Price: Current']) || null;
      const salesRank = parseInt(row['Sales Rank: Current']) || null;
      const reviewRating = parseFloat(row['Reviews: Rating']) || null;
      const reviewCount = parseInt(row['Reviews: Review Count']) || null;

      try {
        // Insert or update the product
        const result = await stmt.run(
          asin, filename, asin, title, brand, description,
          price, listPrice, salesRank, reviewRating, reviewCount
        );

        // Log whether this was an insert or update
        if (existingProduct) {
          console.log(`Updated existing product with ASIN ${asin} from ${filename}`);
          updated++;
        } else {
          console.log(`Inserted new product with ASIN ${asin} from ${filename}`);
          inserted++;
        }

        const productId = result.lastID || existingProduct.id;

        // Define core fields to skip for metadata
        const coreFields = new Set([
          'ASIN', 'asin', 'Title', 'title', 'Brand', 'brand',
          'Description & Features: Description', 'Description & Features: Short Description',
          'Buy Box 🚚: Current', 'List Price: Current', 'Sales Rank: Current',
          'Reviews: Rating', 'Reviews: Review Count'
        ]);

        // Insert product metadata
        let metadataCount = 0;
        for (const [key, value] of Object.entries(row)) {
          if (coreFields.has(key) || value === null || value === undefined) {
            continue;
          }

          await metaStmt.run(productId, key, value.toString());
          metadataCount++;
        }
        
        if (metadataCount > 0) {
          console.log(`Added ${metadataCount} metadata entries for ASIN ${asin}`);
        }
      } catch (error) {
        console.error(`Error inserting/updating product with ASIN ${asin}: ${error.message}`);
        errors++;
      }
    }

    // Finalize prepared statements
    await stmt.finalize();
    await metaStmt.finalize();
    
    // Commit the transaction
    await db.run('COMMIT');
    
    // Log results
    if (duplicatesInBatch.length > 0) {
      console.warn(`Found ${duplicatesInBatch.length} duplicate ASINs within batch from ${filename}`);
    }
    
    console.log(
      `Database batch complete for ${filename}: ` +
      `${inserted} inserted, ${updated} updated, ${skipped} skipped, ` +
      `${duplicatesInBatch.length} duplicates, ${errors} errors`
    );
  } catch (error) {
    console.error(`Transaction failed, rolling back: ${error.message}`);
    try {
      await db.run('ROLLBACK');
      console.log('Transaction rolled back successfully');
    } catch (rollbackError) {
      console.error(`Rollback failed: ${rollbackError.message}`);
    }
    throw error;
  }
}


async function importCSVToDatabase(filePath, filename) {
  return new Promise((resolve, reject) => {
    console.log(`Starting CSV import for ${filename}`);
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      console.warn(`File ${filePath} does not exist, skipping`);
      return resolve();
    }
    
    console.log(`Importing file ${filename} from ${filePath}`);
    
    // Create a read stream for the file
    const readStream = fs.createReadStream(filePath, { encoding: 'utf8' });
    
    // Track processing
    let rowsProcessed = 0;
    let totalRows = 0;
    let batchCount = 0;
    let batchBuffer = [];
    let startTime = Date.now();
    
    // Use PapaParse in streaming mode
    Papa.parse(readStream, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      delimitersToGuess: [',', '\t', '|', ';'],
      
      chunk: async function(results) {
        try {
          // Report parsing errors if any
          if (results.errors && results.errors.length > 0) {
            console.warn(`Parsing warnings in ${filename} chunk: ${JSON.stringify(results.errors)}`);
          }
          
          // Pause the stream to process this chunk completely
          readStream.pause();
          
          const rows = results.data;
          totalRows += rows.length;
          console.log(`Processing chunk of ${rows.length} rows from ${filename}`);
          
          for (const row of rows) {
            const normalizedRow = normalizeRow(row);
            batchBuffer.push(normalizedRow);
            
            // Process in batches
            if (batchBuffer.length >= BATCH_SIZE) {
              batchCount++;
              console.log(`Processing batch ${batchCount} (${batchBuffer.length} rows) from ${filename}`);
              await saveBatchToDatabase(batchBuffer, filename);
              rowsProcessed += batchBuffer.length;
              
              const timeTaken = (Date.now() - startTime) / 1000;
              const rowsPerSecond = Math.round(rowsProcessed / timeTaken);
              console.log(
                `Imported ${rowsProcessed}/${totalRows} rows from ${filename} to database ` +
                `(${Math.round(rowsProcessed/totalRows*100)}%, ${rowsPerSecond} rows/sec)`
              );
              
              batchBuffer = [];
            }
          }
          
          // Resume the stream
          readStream.resume();
        } catch (error) {
          console.error(`Error processing chunk from ${filename}: ${error.message}`);
          readStream.resume(); // Resume even on error
        }
      },
      
      complete: async function() {
        try {
          // Process any remaining rows
          if (batchBuffer.length > 0) {
            batchCount++;
            console.log(`Processing final batch ${batchCount} (${batchBuffer.length} rows) from ${filename}`);
            await saveBatchToDatabase(batchBuffer, filename);
            rowsProcessed += batchBuffer.length;
          }
          
          const timeTaken = (Date.now() - startTime) / 1000;
          const rowsPerSecond = Math.round(rowsProcessed / timeTaken);
          
          console.log(
            `Completed importing ${filename} to database: ` +
            `${rowsProcessed} total rows in ${timeTaken.toFixed(1)} seconds (${rowsPerSecond} rows/sec)`
          );
          
          resolve();
        } catch (error) {
          console.error(`Error completing import of ${filename}: ${error.message}`);
          resolve(); // Resolve anyway to continue with other files
        }
      },
      
      error: function(error) {
        console.error(`Error parsing ${filename}: ${error.message}`);
        resolve(); // Resolve anyway to continue with other files
      }
    });
  });
}

// Check if a CSV file has already been imported to the database
async function checkCSVAlreadyImported(filename) {
  try {
    const result = await db.get(
      'SELECT COUNT(*) as count FROM products WHERE file_source = ?',
      [filename]
    );
    
    const isImported = result && result.count > 0;
    if (isImported) {
      console.log(`CSV file ${filename} has already been imported (${result.count} products found)`);
    } else {
      console.log(`CSV file ${filename} has not been imported yet`);
    }
    
    return isImported;
  } catch (error) {
    console.error(`Error checking if CSV already imported: ${error.message}`);
    return false;
  }
}

/**
 * Vector Store Processing
 */
function preprocessRowsForEmbeddings(rows, filename) {
  return rows.map((row, index) => {
    // Process core text content
    let textContent = '';
    
    // Add title, brand, ASIN
    if (row['Title']) textContent += `Title: ${row['Title']}\n`;
    if (row['Brand']) textContent += `Brand: ${row['Brand']}\n`;
    if (row['ASIN']) textContent += `ASIN: ${row['ASIN']}\n`;
    
    // Add description
    if (row['Description & Features: Description']) {
      textContent += `Description: ${row['Description & Features: Description']}\n`;
    } else if (row['Description & Features: Short Description']) {
      textContent += `Description: ${row['Description & Features: Short Description']}\n`;
    }
    
    // Add features
    const featureFields = [
      'Description & Features: Feature 1', 
      'Description & Features: Feature 2',
      'Description & Features: Feature 3', 
      'Description & Features: Feature 4',
      'Description & Features: Feature 5'
    ];
    
    let featuresText = '';
    for (const field of featureFields) {
      if (row[field]) {
        featuresText += `- ${row[field]}\n`;
      }
    }
    
    if (featuresText) {
      textContent += `Product Features:\n${featuresText}\n`;
    }
    
    // Add sales and ranking data
    if (row['Sales Rank: Current']) {
      textContent += `Current Sales Rank: ${row['Sales Rank: Current']}\n`;
    }
    
    if (row['Reviews: Rating'] && row['Reviews: Review Count']) {
      textContent += `Reviews: ${row['Reviews: Rating']}★ (${row['Reviews: Review Count']} reviews)\n`;
    }
    
    // Add pricing information
    const pricingFields = [
      'Buy Box 🚚: Current', 'List Price: Current', 'Amazon: Current', 
      'New: Current', 'Used: Current'
    ];
    
    let pricingText = 'Pricing Information:\n';
    for (const field of pricingFields) {
      if (row[field]) {
        pricingText += `${field}: ${row[field]}\n`;
      }
    }
    
    if (pricingText.length > 22) { // Length check to avoid empty pricing sections
      textContent += pricingText;
    }
    
    const document = {
      pageContent: textContent,
      metadata: {
        source: filename,
        recordIndex: index,
        asin: row['ASIN'] || 'unknown',
        title: row['Title'] || 'unknown',
        brand: row['Brand'] || 'unknown'
      }
    };
    
    return document;
  });
}

// Process batch with retry logic for OpenAI API calls
async function addEmbeddingBatchWithRetry(vectorStore, batch, filename, batchIndex) {
  await pRetry(
    async () => {
      await vectorStore.addDocuments(batch);
      console.log(`Embedding batch ${batchIndex} from ${filename} succeeded (${batch.length} documents)`);
    },
    {
      retries: 5,
      factor: 2,
      minTimeout: 5000,
      maxTimeout: 60000,
      onFailedAttempt: (error) => {
        console.warn(`Retrying batch ${batchIndex} from ${filename}. Attempt ${error.attemptNumber}. Reason: ${error.message}`);
      }
    }
  );
  // Add delay between batches to avoid rate limiting
  await sleep(2000);
}

// Process CSV files for vector store with controlled concurrency
async function processCSVsForVectorStore() {
  if (isProcessingCSVs) return;
  try {
    isProcessingCSVs = true;
    console.log("Starting vector store creation process...");
    
    // Check if OpenAI API key is available
    if (!process.env.OPENAI_API_KEY) {
      console.error("OpenAI API key not found in environment variables");
      isProcessingCSVs = false;
      return;
    }
    
    // Initialize OpenAI embeddings with configurable parameters
    const embeddings = new OpenAIEmbeddings({ 
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "text-embedding-3-small", // Use smaller, cheaper model
      batchSize: EMBEDDING_BATCH_SIZE, // Process in smaller batches
      maxConcurrency: MAX_CONCURRENT_BATCHES, // Control concurrent API calls
    });
    
    // Create vector store - either in-memory or persistent
    if (USE_PERSISTENT_VECTOR_STORE && HNSWLib) {
      // Ensure directory exists
      if (!fs.existsSync(VECTOR_STORE_PATH)) {
        fs.mkdirSync(VECTOR_STORE_PATH, { recursive: true });
      }
      
      // Create a new HNSWLib instance or load existing one
      try {
        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
        console.log("Loaded existing vector store");
      } catch (error) {
        console.log("Creating new persistent vector store");
        vectorStore = new HNSWLib(embeddings, { space: 'cosine' });
      }
    } else {
      // In-memory vector store (warning: will be lost on server restart)
      console.log(USE_PERSISTENT_VECTOR_STORE && !HNSWLib ? 
        "HNSWLib not available, falling back to in-memory store" : 
        "Using in-memory vector store");
      vectorStore = new MemoryVectorStore(embeddings);
    }
    
    // Process each CSV file using streams
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      if (metadata.recordCount === 0 || metadata.processed) {
        console.warn(`Skipping ${filename} - no records or already processed`);
        continue;
      }
      
      console.log(`Processing ${filename} with ${metadata.recordCount} records for vector store`);
      
      // Create read stream for the file
      const fileStream = fs.createReadStream(metadata.path, 'utf8');
      
      // Process the file
      await new Promise((resolve, reject) => {
        let batchIndex = 0;
        let rowsProcessed = 0;
        let currentBatch = [];
        
        // Stream-based processing
        Papa.parse(fileStream, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          
          chunk: async function(results) {
            try {
              // Pause the stream while processing
              fileStream.pause();
              
              const rows = results.data;
              const documents = preprocessRowsForEmbeddings(rows, filename);
              
              // Process in small batches
              for (const doc of documents) {
                currentBatch.push(doc);
                
                if (currentBatch.length >= EMBEDDING_BATCH_SIZE) {
                  await addEmbeddingBatchWithRetry(vectorStore, currentBatch, filename, batchIndex++);
                  rowsProcessed += currentBatch.length;
                  console.log(`Processed ${rowsProcessed}/${metadata.recordCount} rows from ${filename}`);
                  currentBatch = [];
                }
              }
              
              // Resume the stream
              fileStream.resume();
            } catch (error) {
              console.error(`Error processing chunk from ${filename}:`, error);
              fileStream.resume(); // Resume even on error
            }
          },
          
          complete: async function() {
            try {
              // Process any remaining documents
              if (currentBatch.length > 0) {
                await addEmbeddingBatchWithRetry(vectorStore, currentBatch, filename, batchIndex++);
                rowsProcessed += currentBatch.length;
              }
              
              metadata.processed = true;
              console.log(`Completed processing ${filename}: ${rowsProcessed} records embedded`);
              
              // Save the vector store if using persistent storage
              if (USE_PERSISTENT_VECTOR_STORE && HNSWLib) {
                console.log(`Saving vector store to ${VECTOR_STORE_PATH}`);
                await vectorStore.save(VECTOR_STORE_PATH);
              }
              
              resolve();
            } catch (error) {
              console.error(`Error completing processing of ${filename}:`, error);
              resolve(); // Resolve anyway to continue with other files
            }
          },
          
          error: function(error) {
            console.error(`Error parsing ${filename}:`, error);
            resolve(); // Resolve anyway to continue with other files
          }
        });
      });
    }
    
    console.log("Vector store processing complete!");
    
  } catch (error) {
    console.error('Failed to process CSVs for vector store:', error);
  } finally {
    isProcessingCSVs = false;
  }
}

/**
 * Question Answering with Vector Store
 */
async function getAnswer(question, useGPT4 = false) {
  try {
    // Ensure vector store is loaded
    if (!vectorStore) {
      throw new Error("Vector store not initialized");
    }
    
    // Create the chain with customizations for Amazon product data
    const model = new OpenAI({ 
      temperature: 0.3,  // Lower temperature for more factual responses about products
      modelName: useGPT4 ? "gpt-4" : "gpt-3.5-turbo",  // Select appropriate model
    });
    
    // Configure retriever to get context for the question
    const retriever = vectorStore.asRetriever({
      k: 5,  // Number of documents to retrieve
    });
    
    const chain = await createRetrievalChain({
      retriever: retriever,
      combineDocsChain: null, // Let LangChain handle combining docs
    });

    console.log(`Querying: "${question}"`);
    const result = await chain.invoke({
      input: question,
    });

    // Format the response to include product information
    let response = result.answer || result.output || "";
    
    // Add source information - which products were referenced
    if (result.sourceDocuments && result.sourceDocuments.length > 0) {
      response += "\n\nBased on information from products:\n";
      
      // Create a set of unique ASINs to avoid duplicates
      const mentionedAsins = new Set();
      
      result.sourceDocuments.forEach(doc => {
        const metadata = doc.metadata || {};
        
        if (metadata.asin && !mentionedAsins.has(metadata.asin)) {
          mentionedAsins.add(metadata.asin);
          
          // Add product information
          response += `- ${metadata.title || 'Unknown Product'} (ASIN: ${metadata.asin})`;
          if (metadata.brand && metadata.brand !== 'unknown') {
            response += ` by ${metadata.brand}`;
          }
          response += '\n';
        }
      });
    }

    return response;
  } catch (error) {
    console.error("Error getting answer:", error);
    return `Error processing your question: ${error.message}`;
  }
}

/**
 * API Routes
 */
function setupAPIRoutes() {
  // Get CSV metadata
  app.get('/api/metadata', (req, res) => {
    res.json(csvMetadata);
  });
  
  // Database status endpoint
  app.get('/api/status', async (req, res) => {
    try {
      const productCount = await db.get('SELECT COUNT(*) as count FROM products');
      const fileSourcesResult = await db.all('SELECT DISTINCT file_source FROM products');
      const fileSources = fileSourcesResult.map(row => row.file_source);
      
      res.json({
        status: 'operational',
        database: {
          path: DB_PATH,
          productCount: productCount.count,
          importedFiles: fileSources
        },
        vectorStore: {
          initialized: vectorStore !== null,
          path: VECTOR_STORE_PATH,
          persistent: USE_PERSISTENT_VECTOR_STORE
        }
      });
    } catch (error) {
      console.error('Status check error:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Database query endpoint
  app.post('/api/query', async (req, res) => {
    const { sql, params } = req.body;
    
    try {
      // Limit queries to SELECT statements for security
      if (!sql.trim().toLowerCase().startsWith('select')) {
        return res.status(400).json({ error: 'Only SELECT queries are allowed' });
      }
      
      const results = await db.all(sql, params || []);
      res.json(results);
    } catch (error) {
      console.error('Database query error:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Vector search endpoint
  app.post('/api/vectorsearch', async (req, res) => {
    const { query, limit = 5 } = req.body;
    
    try {
      // Ensure vector store is initialized
      if (!vectorStore) {
        return res.status(503).json({ error: 'Vector store not yet initialized' });
      }
      
      const results = await vectorStore.similaritySearch(query, limit);
      res.json(results);
    } catch (error) {
      console.error('Vector search error:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Enhanced QA endpoint
  app.post('/api/answer', async (req, res) => {
    const { question, useGPT4 = false } = req.body;
    
    try {
      // Ensure vector store is initialized
      if (!vectorStore) {
        return res.status(503).json({ error: 'Vector store not yet initialized' });
      }
      
      const answer = await getAnswer(question, useGPT4);
      res.json({ answer });
    } catch (error) {
      console.error('QA error:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Product search endpoint
  app.get('/api/products', async (req, res) => {
    try {
      const { search, limit = 20, offset = 0 } = req.query;
      
      let sql = 'SELECT * FROM products';
      const params = [];
      
      if (search) {
        sql += ' WHERE title LIKE ? OR asin LIKE ? OR brand LIKE ?';
        params.push(`%${search}%`, `%${search}%`, `%${search}%`);
      }
      
      sql += ' ORDER BY sales_rank LIMIT ? OFFSET ?';
      params.push(parseInt(limit), parseInt(offset));
      
      const results = await db.all(sql, params);
      res.json(results);
    } catch (error) {
      console.error('Product search error:', error);
      res.status(500).json({ error: error.message });
    }
  });
  
  // Product details endpoint
  app.get('/api/products/:asin', async (req, res) => {
    try {
      const { asin } = req.params;
      
      // Get product details
      const product = await db.get('SELECT * FROM products WHERE asin = ?', [asin]);
      
      if (!product) {
        return res.status(404).json({ error: 'Product not found' });
      }
      
      // Get product metadata
      const metadata = await db.all(
        'SELECT key, value FROM product_metadata WHERE product_id = ?',
        [product.id]
      );
      
      // Format metadata as key-value object
      const metadataObj = {};
      metadata.forEach(item => {
        metadataObj[item.key] = item.value;
      });
      
      res.json({
        ...product,
        metadata: metadataObj
      });
    } catch (error) {
      console.error('Product details error:', error);
      res.status(500).json({ error: error.message });
    }
  });
}

/**
 * Server Initialization
 */
async function initServer() {
  try {
    console.log('==== STARTING SERVER INITIALIZATION ====');
    console.log(`Runtime environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`CSV directory: ${CSV_DIR}`);
    console.log(`Vector store path: ${VECTOR_STORE_PATH}`);
    console.log(`Database path: ${DB_PATH}`);
    console.log(`Persistent vector store: ${USE_PERSISTENT_VECTOR_STORE}`);
    
    console.log('Setting up database...');
    await setupDatabase();
    
    console.log('Analyzing CSV files...');
    await analyzeCSVFiles();
    
    console.log('Checking for new CSV files to import...');
    const filesToImport = [];
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      // Skip if already imported
      const alreadyImported = await checkCSVAlreadyImported(filename);
      if (alreadyImported) {
        console.log(`File ${filename} already imported (${metadata.recordCount} records), skipping`);
        continue;
      }
      
      filesToImport.push(filename);
    }
    
    console.log(`Found ${filesToImport.length} new CSV files to import`);
    
    // Import new files
    for (const filename of filesToImport) {
      const metadata = csvMetadata[filename];
      console.log(`Starting import of ${filename} (${metadata.recordCount} records)`);
      await importCSVToDatabase(metadata.path, filename);
    }
    
    console.log('Starting vector store initialization in the background...');
    processCSVsForVectorStore().catch(err => {
      console.error(`Vector store initialization error: ${err.message}`);
    });
    
    
    // Set up API routes
    setupAPIRoutes();
    
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`==== SERVER STARTED SUCCESSFULLY ====`);
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`📁 CSV directory: ${CSV_DIR}`);
      console.log(`💾 Database: ${DB_PATH}`);
      console.log(`🧠 Vector store: ${VECTOR_STORE_PATH}`);
    });
  } catch (error) {
    console.error(`Server initialization error: ${error.message}`);
    process.exit(1);
  }
}
// Start the server
initServer().catch(err => {
  console.error('Failed to start server:', err);
});