// ES Modules format
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

// Option 2: Use the full path to the distribution file
//import { MemoryVectorStore } from '@langchain/community/dist/vectorstores/memory.js';

import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { parse } from 'csv-parse/sync';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
dotenv.config();

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import the vector store module - commented out for now until we convert it
// import { 
//   generateAndStoreEmbeddingsFromCSVs, 
//   processCsvFilesInBatches, 
//   getAnswer 
// } from './vectorStore.js';

const app = express();
app.use(cors());
app.use(express.json());

const CSV_DIR = path.join(__dirname, '../csv');
const VECTOR_STORE_PATH = path.join(__dirname, '../vectorstore');
let keepaData = {};
let vectorStoreInitialized = false;

// Load and parse all CSVs on startup
function loadCSVs() {
  keepaData = {};
  if (!fs.existsSync(CSV_DIR)) return;
  const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
  for (const file of files) {
    const content = fs.readFileSync(path.join(CSV_DIR, file), 'utf8');
    try {
      const records = parse(content, {
          columns: true, 
          bom: true // Add this option to handle the BOM
        });
      keepaData[file] = records;
    } catch (e) {
      console.error(`Failed to parse ${file}:`, e);
    }
  }
}

// Placeholder for the vector store initialization
async function initVectorStore() {
  try {
    console.log('Vector store initialization would happen here');
    // Implementation will be added after we fix import issues
    vectorStoreInitialized = true;
  } catch (error) {
    console.error('Failed to initialize vector store:', error);
  }
}

loadCSVs();

// Initialize vector store without blocking server startup
initVectorStore().catch(err => {
  console.error('Vector store initialization error:', err);
});

app.post('/chat', async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: 'No question provided' });

  // Prepare prompt
  let csvSummary = '';
  for (const [filename, records] of Object.entries(keepaData)) {
    csvSummary += `File: ${filename}\n`;
    csvSummary += JSON.stringify(records.slice(0, 10), null, 2); // Only first 10 rows for brevity
    csvSummary += '\n\n';
  }
  const prompt = `Here's the Keepa data:\n\n${csvSummary}\nUser question: ${question}`;

  // Call OpenAI API (streaming)
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
      stream: true
    })
  });

  res.setHeader('Content-Type', 'text/plain');
  openaiRes.body.pipe(res);
});

// New semantic search API endpoint - placeholder until we fix vector store
app.post('/search', async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: 'No question provided' });
  
  // For testing, just return a simple response
  res.json({ 
    answer: "Vector search functionality will be enabled once import issues are resolved."
  });
});

// Simple test endpoint to verify server is running
app.get('/test', (req, res) => {
  res.json({
    message: 'Server is running!',
    langchainInstalled: true,
    dataLoaded: Object.keys(keepaData).length > 0
  });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
}); 
  
  // Test the imports
  try {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY || "dummy-key",
    });
    console.log("Successfully initialized OpenAIEmbeddings");
    
    // Test memory vector store
    const store = new MemoryVectorStore(embeddings);
    console.log("Successfully initialized MemoryVectorStore");
  } catch (error) {
    console.error("Error testing LangChain imports:", error);
  }


app.get('/', (req, res) => {
  res.send('Welcome to KeepaGPT Backend! Please use one of the following endpoints: /test, /chat, or /search');
});