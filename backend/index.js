// ES Modules format
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import { createReadStream } from 'fs';
import PapaParse from 'papaparse';

dotenv.config();

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const CSV_DIR = path.join(__dirname, '../csv');
const VECTOR_STORE_PATH = path.join(__dirname, '../vectorstore');

// We'll use these to track file metadata without loading all content
let csvMetadata = {};
let vectorStore = null;
let isProcessingCSVs = false;

/**
 * Normalize CSV header keys by trimming whitespace
 * @param {Object} row - CSV row object
 * @returns {Object} - Normalized row with trimmed keys
 */
function normalizeRow(row) {
  const normalizedRow = {};
  for (const [key, value] of Object.entries(row)) {
    const trimmedKey = key.trim();
    normalizedRow[trimmedKey] = value;
  }
  return normalizedRow;
}

/**
 * Get file statistics and metadata without loading full content
 * @returns {Promise<Object>} - File metadata
 */
async function analyzeCSVFiles() {
  return new Promise((resolve, reject) => {
    csvMetadata = {};
    
    if (!fs.existsSync(CSV_DIR)) {
      console.error(`CSV directory does not exist: ${CSV_DIR}`);
      return resolve({}); 
    }

    const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
    console.log(`Found ${files.length} CSV files in directory: ${CSV_DIR}`);

    if (files.length === 0) {
      console.warn(`No CSV files found in directory: ${CSV_DIR}`);
      return resolve({});
    }

    let filesProcessed = 0;
    const totalFiles = files.length;

    // Process each file's metadata
    for (const file of files) {
      const filePath = path.join(CSV_DIR, file);
      console.log(`Analyzing file: ${filePath}`);
      
      if (!fs.existsSync(filePath)) {
        console.error(`File does not exist: ${filePath}`);
        filesProcessed++;
        if (filesProcessed === totalFiles) resolve(csvMetadata);
        continue;
      }

      // Initialize metadata for this file
      csvMetadata[file] = {
        path: filePath,
        recordCount: 0,
        columns: [],
        sampleRows: [],
        processed: false
      };
      
      // Stream parse just enough rows to get metadata and samples
      let rowCount = 0;
      const MAX_SAMPLE_ROWS = 5;
      
      try {
        // Read file content without streaming for initial analysis
        // This is simpler and more reliable than streaming for smaller operations
        const fileContent = fs.readFileSync(filePath, 'utf8');
        
        PapaParse.parse(fileContent, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          delimitersToGuess: [',', '\t', '|', ';'],
          
          step: function(result) {
            if (rowCount === 0 && result.data && typeof result.data === 'object') {
              // First row - capture column headers
              csvMetadata[file].columns = Object.keys(normalizeRow(result.data));
            }
            
            if (rowCount < MAX_SAMPLE_ROWS) {
              // Collect sample rows
              const normalizedRow = normalizeRow(result.data);
              csvMetadata[file].sampleRows.push(normalizedRow);
            }
            
            rowCount++;
          },
          
          complete: function(results) {
            csvMetadata[file].recordCount = results.data.length;
            console.log(`Analyzed ${file}: ${results.data.length} records, ${csvMetadata[file].columns.length} columns`);
            
            filesProcessed++;
            if (filesProcessed === totalFiles) {
              console.log('Completed analyzing all CSV files');
              resolve(csvMetadata);
            }
          },
          
          error: function(error) {
            console.error(`Error analyzing ${file}:`, error);
            filesProcessed++;
            if (filesProcessed === totalFiles) resolve(csvMetadata);
          }
        });
      } catch (err) {
        console.error(`Error reading file ${file}:`, err);
        filesProcessed++;
        if (filesProcessed === totalFiles) resolve(csvMetadata);
      }
    }
  });
}

/**
 * Process CSV files in chunks for the vector store
 * @returns {Promise<void>}
 */
async function processCSVsForVectorStore() {
  if (isProcessingCSVs) {
    console.log('Already processing CSV files, skipping duplicate request');
    return;
  }

  try {
    isProcessingCSVs = true;
    console.log('Starting to process CSV files for vector store...');
    
    // Initialize OpenAI embeddings
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
    
    // Create memory vector store
    vectorStore = new MemoryVectorStore(embeddings);
    
    // Process each file
    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      console.log(`Processing ${filename} with ${metadata.recordCount} records for vector store`);
      
      // Process this file in chunks rather than streaming
      await new Promise((resolve, reject) => {
        const filePath = metadata.path;
        
        try {
          // Use a custom chunk processor to handle large files
          const chunkProcessor = {
            // Add async keyword here to fix the error
            async processFile() {
              try {
                let rowsProcessed = 0;
                const BATCH_SIZE = 100; // Process in batches
                let currentBatch = [];
                
                // Read file in chunks to avoid memory issues
                const fileStream = createReadStream(filePath, {
                  encoding: 'utf8',
                  highWaterMark: 1024 * 1024 // 1MB chunks
                });
                
                let remainder = '';
                let headerLine = '';
                let headers = [];
                
                for await (const chunk of fileStream) {
                  const content = remainder + chunk;
                  const lines = content.split('\n');
                  remainder = lines.pop() || ''; // Keep the last incomplete line
                  
                  // Get headers from first line if not yet processed
                  if (!headerLine && lines.length > 0) {
                    headerLine = lines[0];
                    headers = headerLine.split(',').map(h => h.trim());
                    lines.shift(); // Remove header line
                  }
                  
                  // Process each complete line
                  for (const line of lines) {
                    if (!line.trim()) continue; // Skip empty lines
                    
                    try {
                      // Parse CSV row manually or use PapaParse for single line
                      const parsed = PapaParse.parse(line, {
                        header: false,
                        skipEmptyLines: true,
                        dynamicTyping: true,
                      }).data[0];
                      
                      // Create row object from headers and values
                      const row = {};
                      for (let i = 0; i < Math.min(headers.length, parsed.length); i++) {
                        row[headers[i]] = parsed[i];
                      }
                      
                      // Normalize the row
                      const normalizedRow = normalizeRow(row);
                      rowsProcessed++;
                      
                      // Create document for vector store
                      const textContent = Object.entries(normalizedRow)
                        .filter(([_, value]) => value !== null && value !== undefined)
                        .map(([key, value]) => `${key}: ${value}`)
                        .join('\n');
                      
                      const document = {
                        pageContent: textContent,
                        metadata: {
                          source: filename,
                          recordIndex: rowsProcessed,
                          id: normalizedRow.asin || normalizedRow.ASIN || `record-${rowsProcessed}`
                        }
                      };
                      
                      currentBatch.push(document);
                      
                      // When batch is full, add to vector store and clear
                      if (currentBatch.length >= BATCH_SIZE) {
                        await vectorStore.addDocuments(currentBatch);
                        console.log(`Added batch of ${currentBatch.length} documents from ${filename} (${rowsProcessed}/${metadata.recordCount})`);
                        
                        currentBatch = [];
                      }
                      
                      // Log progress for large files
                      if (rowsProcessed % 1000 === 0) {
                        console.log(`${filename} progress: ${rowsProcessed}/${metadata.recordCount} records`);
                      }
                    } catch (err) {
                      console.error(`Error processing row in ${filename}:`, err);
                      // Continue despite errors in individual rows
                    }
                  }
                }
                
                // Process any remaining content in the last chunk
                if (remainder.trim()) {
                  try {
                    const parsed = PapaParse.parse(remainder, {
                      header: false,
                      skipEmptyLines: true,
                      dynamicTyping: true,
                    }).data[0];
                    
                    // Create row object from headers and values
                    const row = {};
                    for (let i = 0; i < Math.min(headers.length, parsed.length); i++) {
                      row[headers[i]] = parsed[i];
                    }
                    
                    // Normalize and add to batch
                    const normalizedRow = normalizeRow(row);
                    rowsProcessed++;
                    
                    const textContent = Object.entries(normalizedRow)
                      .filter(([_, value]) => value !== null && value !== undefined)
                      .map(([key, value]) => `${key}: ${value}`)
                      .join('\n');
                    
                    currentBatch.push({
                      pageContent: textContent,
                      metadata: {
                        source: filename,
                        recordIndex: rowsProcessed,
                        id: normalizedRow.asin || normalizedRow.ASIN || `record-${rowsProcessed}`
                      }
                    });
                  } catch (err) {
                    console.error(`Error processing final row in ${filename}:`, err);
                  }
                }
                
                // Add any remaining documents in the final batch
                if (currentBatch.length > 0) {
                  await vectorStore.addDocuments(currentBatch);
                  console.log(`Added final batch of ${currentBatch.length} documents from ${filename}`);
                }
                
                console.log(`Completed processing ${filename}: ${rowsProcessed} records`);
                metadata.processed = true;
              } catch (error) {
                console.error(`Error in chunk processing for ${filename}:`, error);
              }
            }
          };
          
          // Alternative approach using PapaParse's complete parsing for smaller files
          const useFastApproach = metadata.recordCount < 10000;
          
          if (useFastApproach) {
            console.log(`Using fast approach for ${filename} (${metadata.recordCount} records)`);
            // For smaller files, we can use the complete parsing approach
            const fileContent = fs.readFileSync(filePath, 'utf8');
            
            PapaParse.parse(fileContent, {
              header: true,
              skipEmptyLines: true,
              dynamicTyping: true,
              delimitersToGuess: [',', '\t', '|', ';'],
              
              complete: async function(results) {
                try {
                  const BATCH_SIZE = 100;
                  let currentBatch = [];
                  
                  for (let i = 0; i < results.data.length; i++) {
                    const normalizedRow = normalizeRow(results.data[i]);
                    
                    // Create document for vector store
                    const textContent = Object.entries(normalizedRow)
                      .filter(([_, value]) => value !== null && value !== undefined)
                      .map(([key, value]) => `${key}: ${value}`)
                      .join('\n');
                    
                    const document = {
                      pageContent: textContent,
                      metadata: {
                        source: filename,
                        recordIndex: i,
                        id: normalizedRow.asin || normalizedRow.ASIN || `record-${i}`
                      }
                    };
                    
                    currentBatch.push(document);
                    
                    // When batch is full, add to vector store and clear
                    if (currentBatch.length >= BATCH_SIZE || i === results.data.length - 1) {
                      await vectorStore.addDocuments(currentBatch);
                      console.log(`Added batch of ${currentBatch.length} documents from ${filename} (${i+1}/${results.data.length})`);
                      currentBatch = [];
                    }
                    
                    // Log progress for larger files
                    if (i > 0 && i % 1000 === 0) {
                      console.log(`${filename} progress: ${i}/${results.data.length} records`);
                    }
                  }
                  
                  console.log(`Completed processing ${filename}: ${results.data.length} records`);
                  metadata.processed = true;
                  resolve();
                } catch (error) {
                  console.error(`Error processing file ${filename}:`, error);
                  resolve(); // Continue with other files
                }
              },
              
              error: function(error) {
                console.error(`Error parsing ${filename}:`, error);
                resolve(); // Continue with other files
              }
            });
          } else {
            // For larger files, use the chunk processor
            console.log(`Using chunk processor for large file ${filename} (${metadata.recordCount} records)`);
            chunkProcessor.processFile().then(() => {
              resolve();
            }).catch(err => {
              console.error(`Error processing file ${filename} with chunk processor:`, err);
              resolve(); // Continue with other files
            });
          }
        } catch (err) {
          console.error(`Error reading file ${filename}:`, err);
          resolve(); // Continue with other files
          resolve(); // Continue with other files
        }
      });
    }
    
    console.log('Vector store processing complete');
  } catch (error) {
    console.error('Failed to process CSVs for vector store:', error);
  } finally {
    isProcessingCSVs = false;
  }
}

/**
 * Get CSV data sample for a specific query
 * This streams through files to find relevant data without loading everything
 * @param {string} question - User question
 * @returns {Promise<Object>} - Relevant data samples
 */
async function getRelevantDataSample(question) {
  // For simplicity, return file metadata and samples
  const sampleData = {};
  
  for (const [filename, metadata] of Object.entries(csvMetadata)) {
    sampleData[filename] = {
      recordCount: metadata.recordCount,
      columns: metadata.columns,
      sampleRows: metadata.sampleRows
    };
  }
  
  return sampleData;
}

// Initialize the server
async function initServer() {
  try {
    // Analyze CSV metadata first (without loading full content)
    console.log('Analyzing CSV files...');
    await analyzeCSVFiles();
    
    // Start vector store processing in the background
    console.log('Starting vector store initialization in the background...');
    processCSVsForVectorStore().catch(err => {
      console.error('Vector store initialization error:', err);
    });
    
    // Start the server
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`Backend listening on port ${PORT}`);
      console.log(`CSV directory path: ${CSV_DIR}`);
    });
  } catch (error) {
    console.error('Server initialization error:', error);
    process.exit(1);
  }
}

// Improved chat endpoint using streaming approach
app.post('/chat', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: 'No question provided' });
    }

    // Check if we have data to work with
    if (Object.keys(csvMetadata).length === 0) {
      return res.status(500).json({ 
        error: 'No CSV data available. Please ensure CSV files are properly loaded.'
      });
    }

    // Get relevant data sample for the question
    const relevantData = await getRelevantDataSample(question);
    
    // Prepare prompt with CSV data representation
    let csvSummary = '';
    for (const [filename, data] of Object.entries(relevantData)) {
      csvSummary += `File: ${filename}\n`;
      csvSummary += `Total Records: ${data.recordCount}\n`;
      csvSummary += `Columns: ${data.columns.join(', ')}\n`;
      
      // Add sample data
      csvSummary += `Sample data (${Math.min(3, data.sampleRows.length)} records):\n`;
      csvSummary += JSON.stringify(data.sampleRows.slice(0, 3), null, 2);
      csvSummary += '\n\n';
    }
    
    const prompt = `
You are an expert in analyzing Amazon Keepa data. Please analyze the following data and answer the user's question.

CSV Data Summary:
${csvSummary}

User question: ${question}

Provide a detailed answer based on the data, including any relevant statistics or trends you can identify.
`;

    // Call OpenAI API with improved error handling
    const openaiRes = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: 'You are a helpful assistant for analyzing Amazon Keepa CSV data.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.3, // Lower temperature for more factual responses
        stream: false
      })
    });

    // Check if response is ok
    if (!openaiRes.ok) {
      const errorText = await openaiRes.text();
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch (e) {
        errorData = { raw: errorText };
      }
      console.error('OpenAI API error:', errorData);
      return res.status(openaiRes.status).json({ 
        error: 'Error from OpenAI API', 
        details: errorData
      });
    }

    // Parse and return the JSON response
    const data = await openaiRes.json();
    return res.json({
      answer: data.choices[0].message.content
    });
    
  } catch (error) {
    console.error('Error in /chat endpoint:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message 
    });
  }
});

// Vector search API endpoint
app.post('/search', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: 'No question provided' });
    }
    
    // Check if vector store is initialized
    if (!vectorStore) {
      return res.status(503).json({ 
        error: 'Vector store not yet initialized. Please try again later.' 
      });
    }
    
    // Perform vector search
    const results = await vectorStore.similaritySearch(question, 5);
    
    return res.json({ 
      results: results.map(doc => ({
        content: doc.pageContent,
        metadata: doc.metadata
      })),
      dataStats: {
        files: Object.keys(csvMetadata).length,
        totalRecords: Object.values(csvMetadata).reduce((sum, data) => sum + data.recordCount, 0)
      }
    });
  } catch (error) {
    console.error('Error in /search endpoint:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message 
    });
  }
});

// Updated endpoint to check CSV data
app.get('/csv-status', (req, res) => {
  try {
    const fileStats = Object.entries(csvMetadata).map(([filename, data]) => ({
      filename,
      recordCount: data.recordCount,
      sampleFields: data.columns.slice(0, 10),
      processed: data.processed,
      hasData: data.recordCount > 0
    }));
    
    res.json({
      totalFiles: Object.keys(csvMetadata).length,
      totalRecords: Object.values(csvMetadata).reduce((sum, data) => sum + data.recordCount, 0),
      vectorStoreReady: vectorStore !== null,
      isCurrentlyProcessing: isProcessingCSVs,
      files: fileStats
    });
  } catch (error) {
    console.error('Error in /csv-status endpoint:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message 
    });
  }
});

// Simple test endpoint to verify server is running
app.get('/test', (req, res) => {
  try {
    res.json({
      message: 'Server is running!',
      langchainInstalled: true,
      dataAnalyzed: Object.keys(csvMetadata).length > 0,
      csvFiles: Object.keys(csvMetadata),
      recordCounts: Object.fromEntries(
        Object.entries(csvMetadata).map(([file, data]) => [file, data.recordCount])
      ),
      processingStatus: isProcessingCSVs ? 'Processing CSVs' : 'Idle'
    });
  } catch (error) {
    console.error('Error in /test endpoint:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message 
    });
  }
});

// Endpoint to trigger CSV reprocessing
app.post('/reprocess-csvs', (req, res) => {
  if (isProcessingCSVs) {
    return res.status(409).json({
      message: 'Already processing CSV files',
      status: 'processing'
    });
  }
  
  // Start processing in background
  processCSVsForVectorStore().catch(err => {
    console.error('Vector store processing error:', err);
  });
  
  res.json({
    message: 'Started CSV processing',
    status: 'started'
  });
});

app.get('/', (req, res) => {
  res.send('Welcome to KeepaGPT Backend! Please use one of the following endpoints: /test, /chat, /search, or /csv-status');
});

// Start the server
initServer().catch(err => {
  console.error('Failed to start server:', err);
});