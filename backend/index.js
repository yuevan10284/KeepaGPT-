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
import pRetry from 'p-retry';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const CSV_DIR = path.join(__dirname, '../csv');
const VECTOR_STORE_PATH = path.join(__dirname, '../vectorstore');

let csvMetadata = {};
let vectorStore = null;
let isProcessingCSVs = false;

function normalizeRow(row) {
  const normalizedRow = {};
  for (const [key, value] of Object.entries(row)) {
    const trimmedKey = key.trim();
    normalizedRow[trimmedKey] = value;
  }
  return normalizedRow;
}

async function analyzeCSVFiles() {
  return new Promise((resolve, reject) => {
    csvMetadata = {};
    if (!fs.existsSync(CSV_DIR)) return resolve({});
    const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
    if (files.length === 0) return resolve({});
    let filesProcessed = 0;
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
        const fileContent = fs.readFileSync(filePath, 'utf8');
        const results = PapaParse.parse(fileContent, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true
        });
        const normalized = results.data.map(normalizeRow);
        csvMetadata[file].columns = Object.keys(normalized[0] || {});
        csvMetadata[file].sampleRows = normalized.slice(0, 5);
        csvMetadata[file].recordCount = normalized.length;
      } catch (err) {
        console.error(`Error reading file ${file}:`, err);
      }
      filesProcessed++;
      if (filesProcessed === files.length) resolve(csvMetadata);
    }
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function addBatchWithRetry(batch, filename, batchIndex) {
  await pRetry(
    async () => {
      await vectorStore.addDocuments(batch);
      console.log(`Batch ${batchIndex} from ${filename} succeeded`);
    },
    {
      retries: 10,
      factor: 2,
      minTimeout: 5000,
      maxTimeout: 60000,
      onFailedAttempt: (error) => {
        console.warn(`Retrying batch ${batchIndex} from ${filename}. Attempt ${error.attemptNumber}. Reason: ${error.message}`);
      }
    }
  );
  await sleep(10000); // wait 10 seconds after each successful batch
}

async function processCSVsForVectorStore() {
  if (isProcessingCSVs) return;
  try {
    isProcessingCSVs = true;
    const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
    vectorStore = new MemoryVectorStore(embeddings);

    for (const [filename, metadata] of Object.entries(csvMetadata)) {
      if (metadata.recordCount === 0) {
        console.warn(`Skipping ${filename} - no records`);
        continue;
      }

      console.log(`Processing ${filename} with ${metadata.recordCount} records for vector store`);

      await new Promise((resolve) => {
        PapaParse.parse(fs.readFileSync(metadata.path, 'utf8'), {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          complete: async function (results) {
            try {
              const BATCH_SIZE = 25;
              let currentBatch = [];
              let batchIndex = 0;

              for (let i = 0; i < results.data.length; i++) {
                const normalizedRow = normalizeRow(results.data[i]);
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

                if (currentBatch.length >= BATCH_SIZE || i === results.data.length - 1) {
                  await addBatchWithRetry(currentBatch, filename, batchIndex++);
                  currentBatch = [];
                }
              }

              metadata.processed = true;
              console.log(`Completed processing ${filename}: ${results.data.length} records`);
              resolve();
            } catch (error) {
              console.error(`Error processing file ${filename}:`, error);
              resolve();
            }
          },
          error: function (error) {
            console.error(`Error parsing ${filename}:`, error);
            resolve();
          }
        });
      });
    }

    console.log('Vector store processing complete');
  } catch (error) {
    console.error('Failed to process CSVs for vector store:', error);
  } finally {
    isProcessingCSVs = false;
  }
}

async function initServer() {
  try {
    console.log('Analyzing CSV files...');
    await analyzeCSVFiles();
    console.log('Starting vector store initialization in the background...');
    processCSVsForVectorStore().catch(err => {
      console.error('Vector store initialization error:', err);
    });
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

initServer().catch(err => {
  console.error('Failed to start server:', err);
});
