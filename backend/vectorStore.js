// ES Module format
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/text-splitter";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import fs from "fs";
import path, { dirname } from "path";
import { RetrievalQAChain, loadQARefineChain } from "@langchain/core/chains";
import { fileURLToPath } from 'url';
import PapaParse from 'papaparse';

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Processes multiple CSV files from a directory and creates embeddings
 * @param {string} csvDirectory - Path to directory containing CSV files
 * @param {string} outputPath - Path to save the vector store
 * @param {number} chunkSize - Size of text chunks for processing
 * @param {number} chunkOverlap - Overlap between chunks
 */
export async function generateAndStoreEmbeddingsFromCSVs(
  csvDirectory = "./csv_files",
  outputPath = "hnswlib", 
  chunkSize = 1000,
  chunkOverlap = 200
) {
  console.log(`Processing CSV files from: ${csvDirectory}`);
  
  // STEP 1: Get all CSV files from the directory
  const files = fs.readdirSync(csvDirectory)
    .filter(file => file.toLowerCase().endsWith('.csv'))
    .map(file => path.join(csvDirectory, file));
  
  if (files.length === 0) {
    console.error("No CSV files found in the specified directory");
    return;
  }
  
  console.log(`Found ${files.length} CSV files: ${files.join(", ")}`);
  
  // STEP 2: Initialize text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });
  
  // STEP 3: Process each CSV file and collect documents
  let allDocs = [];
  
  for (const file of files) {
    try {
      console.log(`Processing file: ${file}`);
      
      // Parse CSV with PapaParse for better control
      const csvText = fs.readFileSync(file, 'utf8');
      const { data, meta } = PapaParse.parse(csvText, {
        header: true,
        skipEmptyLines: 'greedy',
        dynamicTyping: true,
        // For this specific dataset, we need to handle lots of columns
        transformHeader: (header) => header.trim()
      });
      
      console.log(`Parsed ${data.length} rows from ${file}`);
      
      // Process the products data
      const productsText = data.map(row => {
        // Start with product identifiers and critical information
        let productText = '';
        
        // Add core product details first for better context
        const coreFields = [
          'Title', 'Brand', 'ASIN', 'Description & Features: Description', 
          'Description & Features: Short Description', 'Parent Title', 'Manufacturer'
        ];
        
        for (const field of coreFields) {
          if (row[field]) {
            productText += `${field}: ${row[field]}\n`;
          }
        }
        
        // Add features separately for better organization
        const featureFields = [
          'Description & Features: Feature 1', 'Description & Features: Feature 2',
          'Description & Features: Feature 3', 'Description & Features: Feature 4',
          'Description & Features: Feature 5', 'Description & Features: Feature 6',
          'Description & Features: Feature 7', 'Description & Features: Feature 8',
          'Description & Features: Feature 9', 'Description & Features: Feature 10'
        ];
        
        let featuresText = '';
        for (const field of featureFields) {
          if (row[field]) {
            featuresText += `- ${row[field]}\n`;
          }
        }
        
        if (featuresText) {
          productText += `Product Features:\n${featuresText}\n`;
        }
        
        // Add sales and ranking data
        if (row['Sales Rank: Current']) {
          productText += `Current Sales Rank: ${row['Sales Rank: Current']}\n`;
        }
        
        if (row['Reviews: Rating'] && row['Reviews: Review Count']) {
          productText += `Reviews: ${row['Reviews: Rating']}★ (${row['Reviews: Review Count']} reviews)\n`;
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
          productText += pricingText;
        }
        
        // Add remaining fields that aren't covered above (limiting to avoid overly large documents)
        const importantRemainingFields = [
          'Product Group', 'Model', 'Categories: Root', 'Categories: Sub', 
          'Format', 'Author', 'Contributors', 'Publication Date', 'Release Date'
        ];
        
        for (const field of importantRemainingFields) {
          if (row[field]) {
            productText += `${field}: ${row[field]}\n`;
          }
        }
        
        return productText;
      });
      
      // Create documents with metadata
      const docs = await textSplitter.createDocuments(
        productsText,
        data.map(row => ({ 
          source: file,
          asin: row['ASIN'] || 'unknown',
          title: row['Title'] || 'unknown',
          brand: row['Brand'] || 'unknown',
          productGroup: row['Product Group'] || 'unknown'
        }))
      );
      
      console.log(`Created ${docs.length} document chunks from ${file}`);
      allDocs = allDocs.concat(docs);
    } catch (error) {
      console.error(`Error processing file ${file}:`, error);
    }
  }
  
  console.log(`Total document chunks across all files: ${allDocs.length}`);
  
  if (allDocs.length === 0) {
    console.error("No documents were extracted from the CSV files");
    return;
  }
  
  // STEP 4: Generate embeddings from all documents
  console.log("Generating embeddings...");
  const vectorStore = await HNSWLib.fromDocuments(
    allDocs,
    new OpenAIEmbeddings(),
  );
  
  // STEP 5: Save the vector store
  console.log(`Saving vector store to: ${outputPath}`);
  await vectorStore.save(outputPath);
  
  console.log("Embedding generation complete!");
}

// Function to handle large CSV files by processing them in batches
export async function processCsvFilesInBatches(
  csvDirectory = "./csv_files",
  outputPath = "hnswlib",
  batchSize = 1000, // Number of rows per batch
  chunkSize = 1000,
  chunkOverlap = 200
) {
  console.log(`Processing large CSV files in batches from: ${csvDirectory}`);
  
  // Get all CSV files
  const files = fs.readdirSync(csvDirectory)
    .filter(file => file.toLowerCase().endsWith('.csv'))
    .map(file => path.join(csvDirectory, file));
  
  if (files.length === 0) {
    console.error("No CSV files found in the specified directory");
    return;
  }
  
  // Initialize embeddings model
  const embeddings = new OpenAIEmbeddings();
  
  // Initialize text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });
  
  // Process each file
  let vectorStore = null;
  
  for (const file of files) {
    try {
      console.log(`Processing file: ${file}`);
      
      // Use a CSV parsing library that supports streams
      const fileStream = fs.createReadStream(file);
      
      let batch = [];
      let batchNumber = 0;
      let header = null;
      
      // Process the file in streaming mode
      await new Promise((resolve, reject) => {
        PapaParse.parse(fileStream, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: 'greedy',  // Handle various types of empty lines
          transformHeader: (header) => header.trim(), // Trim whitespace from headers
          delimitersToGuess: [',', '\t', '|', ';'], // Try to guess delimiter if not comma
          step: async function(row) {
            if (!header) {
              header = Object.keys(row.data).map(key => key.trim());
            }
            
            // Handle Amazon product data specifically
            const data = row.data;
            
            // Skip rows with no essential product data (likely malformed rows)
            if (!data['ASIN'] && !data['Title']) {
              return;
            }
            
            batch.push(data);
            
            // When batch is full, process it
            if (batch.length >= batchSize) {
              // Pause the parser while we process the batch
              fileStream.pause();
              
              try {
                await processDocumentBatch(
                  batch, 
                  batchNumber, 
                  file,
                  textSplitter, 
                  embeddings, 
                  vectorStore
                ).then(result => {
                  vectorStore = result;
                });
                
                batchNumber++;
                batch = [];
                
                // Resume parsing
                fileStream.resume();
              } catch (error) {
                console.error(`Error processing batch ${batchNumber}:`, error);
                // Continue processing despite batch errors
                batchNumber++;
                batch = [];
                fileStream.resume();
              }
            }
          },
          complete: async function() {
            // Process any remaining records
            if (batch.length > 0) {
              try {
                await processDocumentBatch(
                  batch, 
                  batchNumber, 
                  file,
                  textSplitter, 
                  embeddings, 
                  vectorStore
                ).then(result => {
                  vectorStore = result;
                });
                
                resolve();
              } catch (error) {
                console.error(`Error processing final batch:`, error);
                resolve(); // Resolve anyway to continue with other files
              }
            } else {
              resolve();
            }
          },
          error: function(error) {
            console.error(`Error parsing file ${file}:`, error);
            resolve(); // Resolve anyway to continue with other files
          }
        });
      });
      
      console.log(`Finished processing file: ${file}`);
      
    } catch (error) {
      console.error(`Error processing file ${file}:`, error);
    }
  }
  
  // Save the final vector store
  if (vectorStore) {
    console.log(`Saving final vector store to: ${outputPath}`);
    await vectorStore.save(outputPath);
  } else {
    console.error("No vector store was created. Check if any documents were successfully processed.");
  }
  
  console.log("Large CSV processing complete!");
}

// Helper function to process a batch of CSV rows
async function processDocumentBatch(batch, batchNumber, fileName, textSplitter, embeddings, existingVectorStore) {
  console.log(`Processing batch ${batchNumber} from ${fileName} with ${batch.length} rows`);
  
  // Convert batch to text documents - with special handling for Amazon product data
  const batchText = batch.map(row => {
    // Start with product identifiers and critical information
    let productText = '';
    
    // Add core product details first for better context
    const coreFields = [
      'Title', 'Brand', 'ASIN', 'Description & Features: Description', 
      'Description & Features: Short Description', 'Parent Title', 'Manufacturer'
    ];
    
    for (const field of coreFields) {
      if (row[field]) {
        productText += `${field}: ${row[field]}\n`;
      }
    }
    
    // Add features separately for better organization
    const featureFields = [
      'Description & Features: Feature 1', 'Description & Features: Feature 2',
      'Description & Features: Feature 3', 'Description & Features: Feature 4',
      'Description & Features: Feature 5', 'Description & Features: Feature 6',
      'Description & Features: Feature 7', 'Description & Features: Feature 8',
      'Description & Features: Feature 9', 'Description & Features: Feature 10'
    ];
    
    let featuresText = '';
    for (const field of featureFields) {
      if (row[field]) {
        featuresText += `- ${row[field]}\n`;
      }
    }
    
    if (featuresText) {
      productText += `Product Features:\n${featuresText}\n`;
    }
    
    // Add sales and ranking data
    if (row['Sales Rank: Current']) {
      productText += `Current Sales Rank: ${row['Sales Rank: Current']}\n`;
    }
    
    if (row['Reviews: Rating'] && row['Reviews: Review Count']) {
      productText += `Reviews: ${row['Reviews: Rating']}★ (${row['Reviews: Review Count']} reviews)\n`;
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
      productText += pricingText;
    }
    
    // Add remaining fields that aren't covered above
    for (const [key, value] of Object.entries(row)) {
      // Skip fields we've already added
      if (coreFields.includes(key) || featureFields.includes(key) || 
          pricingFields.includes(key) || 
          key === 'Sales Rank: Current' || 
          key === 'Reviews: Rating' || 
          key === 'Reviews: Review Count') {
        continue;
      }
      
      // Skip empty values
      if (value === null || value === undefined || value === '') {
        continue;
      }
      
      // Add the remaining fields
      productText += `${key}: ${value}\n`;
    }
    
    return productText;
  });
  
  // Create documents with metadata about the product and source
  const docs = await textSplitter.createDocuments(
    batchText,
    batch.map((row, i) => ({ 
      source: `${fileName}-batch${batchNumber}-row${i}`,
      asin: row['ASIN'] || 'unknown',
      title: row['Title'] || 'unknown',
      brand: row['Brand'] || 'unknown'
    }))
  );
  
  console.log(`Created ${docs.length} document chunks from batch`);
  
  // Add to vector store
  let vectorStore;
  if (existingVectorStore) {
    // Add to existing vector store
    await existingVectorStore.addDocuments(docs);
    vectorStore = existingVectorStore;
  } else {
    // Create new vector store
    vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
  }
  
  return vectorStore;
}

// Updated query function to work with the vector store
export async function getAnswer(question, vectorStorePath = "hnswlib") {
  console.log(`Loading vector store from: ${vectorStorePath}`);
  
  // STEP 1: Load the vector store
  const vectorStore = await HNSWLib.load(
    vectorStorePath,
    new OpenAIEmbeddings(),
  );

  // STEP 2: Create the chain with customizations for Amazon product data
  const model = new OpenAI({ 
    temperature: 0.3,  // Lower temperature for more factual responses about products
    modelName: "gpt-4",  // Using a more capable model for better understanding of product data
  });
  
  // Configure retriever to get more context for the question
  const retriever = vectorStore.asRetriever({
    k: 5,  // Retrieve more documents for better context
  });
  
  const chain = new RetrievalQAChain({
    combineDocumentsChain: loadQARefineChain(model),
    retriever: retriever,
    returnSourceDocuments: true,  // Include source documents in the response
  });

  // STEP 3: Get the answer
  console.log(`Querying: "${question}"`);
  const result = await chain.call({
    query: question,
  });

  // STEP 4: Format the response to include product information
  let response = result.text || result.output_text;
  
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
}