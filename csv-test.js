import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import assert from 'assert';
import PapaParse from 'papaparse';

// Fix for __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to the CSV files directory
const CSV_DIR = path.join(__dirname, 'csv');

// Function to test parsing of all CSV files in the directory
function testAllCSVFiles() {
  console.log("Starting CSV parsing tests...");
  
  // Get all CSV files in the directory
  const files = fs.readdirSync(CSV_DIR).filter(file => file.endsWith('.csv'));
  
  if (files.length === 0) {
    console.log("No CSV files found in directory:", CSV_DIR);
    return;
  }
  
  console.log(`Found ${files.length} CSV files to test`);
  
  // Test each file
  for (const file of files) {
    testCSVFile(file);
  }
  
  console.log("\nAll CSV parsing tests completed successfully!");
}

// Function to test a single CSV file
function testCSVFile(filename) {
  const filePath = path.join(CSV_DIR, filename);
  console.log(`\nTesting CSV file: ${filename}`);
  
  try {
    // Read and parse the file
    const csvText = fs.readFileSync(filePath, 'utf8');
    const result = PapaParse.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      delimitersToGuess: [',', '\t', '|', ';']
    });
    
    // Basic validation
    assert.ok(result.data.length > 0, `File ${filename} should have at least one row of data`);
    assert.ok(result.meta.fields.length > 0, `File ${filename} should have at least one column`);
    
    // Headers info
    console.log(`Headers: ${result.meta.fields.join(', ')}`);
    console.log(`Total rows: ${result.data.length}`);
    
    // Check the first 5 rows (or all rows if less than 5)
    const rowsToCheck = Math.min(5, result.data.length);
    console.log(`Validating first ${rowsToCheck} rows:`);
    
    for (let i = 0; i < rowsToCheck; i++) {
      const row = result.data[i];
      
      // Verify row is an object with data
      assert.ok(typeof row === 'object' && row !== null, `Row ${i} should be a valid object`);
      
      // Verify each field in the row has a corresponding header
      for (const key in row) {
        assert.ok(
          result.meta.fields.includes(key),
          `Field "${key}" in row ${i} should match a header column`
        );
      }
      
      // Output first row for visibility
      if (i === 0) {
        console.log("First row sample:", JSON.stringify(row, null, 2));
      }
    }
    
    console.log(`✓ CSV parsing test for ${filename} passed!`);
    
    // Test product text extraction for the first row
    if (result.data.length > 0) {
      const productText = createProductText(result.data[0]);
      console.log("\nProduct text extraction sample:");
      console.log(productText);
    }
    
    return true;
  } catch (error) {
    console.error(`Test failed for ${filename}:`, error);
    throw error;
  }
}

// Simple function to extract product text
function createProductText(row) {
  let productText = '';
  
  // Try to identify common product fields dynamically
  const possibleFields = [
    'Title', 'Brand', 'ASIN', 'Description', 'Features',
    'Description & Features: Description', 'Product Description',
    'Price', 'Category', 'SKU'
  ];
  
  // Add any fields that exist in the row
  for (const field of possibleFields) {
    if (row[field] !== undefined && row[field] !== null) {
      productText += `${field}: ${row[field]}\n`;
    }
  }
  
  // If no recognized fields were found, include all fields
  if (productText.trim() === '') {
    for (const [key, value] of Object.entries(row)) {
      if (value !== undefined && value !== null) {
        productText += `${key}: ${value}\n`;
      }
    }
  }
  
  return productText;
}

// Run the original sample test as well for backward compatibility
function runOriginalTest() {
  console.log("\n=== Running original sample test ===");
  
  // Sample data for testing
  const mockCsvData = `Title,Brand,ASIN,Description & Features: Description
"Test Product","Test Brand","B00TEST123","This is a test product description"
"Test Product 2","Brand 2","B00TEST456","Another description"`;
  
  // Create a temporary file
  const tempDir = path.join(__dirname, 'temp_test_dir');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  const testFile = path.join(tempDir, 'test.csv');
  fs.writeFileSync(testFile, mockCsvData);
  
  // Read and parse the file
  try {
    const csvText = fs.readFileSync(testFile, 'utf8');
    const result = PapaParse.parse(csvText, {
      header: true,
      skipEmptyLines: true
    });
    
    // Verify parsing results
    console.log("CSV parsing test results:");
    console.log("Number of rows:", result.data.length);
    console.log("First row:", result.data[0]);
    
    // Basic assertions
    assert.strictEqual(result.data.length, 2, "Should have 2 rows");
    assert.strictEqual(result.data[0].ASIN, "B00TEST123", "First ASIN should match");
    assert.strictEqual(result.data[1].ASIN, "B00TEST456", "Second ASIN should match");
    
    console.log("✓ Original CSV parsing test passed!");
    
    // Test product text extraction
    const productText = createProductText(result.data[0]);
    console.log("\nProduct text extraction:");
    console.log(productText);
    
    // Clean up
    fs.unlinkSync(testFile);
    fs.rmdirSync(tempDir);
    
  } catch (error) {
    console.error("Original test failed:", error);
    throw error;
  }
}

// Main function to run all tests
function runAllTests() {
  // First test all CSV files
  testAllCSVFiles();
  
  // Then run the original test for compatibility
  runOriginalTest();
}

// Run all tests
runAllTests();