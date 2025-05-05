export default async function handler(req, res) {
  try {
    // Set up streaming response headers
    res.setHeader("Content-Type", "text/plain");
    
    // Forward the request to the backend
    const backendRes = await fetch("http://localhost:5000/api/vectorsearch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ 
        query: req.body.query || req.body.question // Support both query formats
      }),
    });

    if (!backendRes.ok) {
      throw new Error(`Backend returned status: ${backendRes.status}`);
    }

    // Stream the backend response to the frontend
    const reader = backendRes.body.getReader();
    const decoder = new TextDecoder();
    
    // Process chunks of data from backend
    let buffer = "";
    let productData = [];
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      // Add new chunk to buffer and process
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines in buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ""; // Keep the last incomplete chunk in buffer
      
      // Process each complete line
      lines.forEach(line => {
        if (line.trim()) {
          try {
            const product = JSON.parse(line);
            productData.push(product);
          } catch (e) {
            console.error("Error parsing product data:", e);
          }
        }
      });
    }
    
    // Process any remaining data in buffer
    if (buffer.trim()) {
      try {
        const product = JSON.parse(buffer);
        productData.push(product);
      } catch (e) {
        console.error("Error parsing final product data:", e);
      }
    }
    
    // Generate a response based on the product data
    const response = formatProductDataResponse(req.body.query || req.body.question, productData);
    
    // Send the formatted response
    res.write(response);
    res.end();
  } catch (error) {
    console.error("API route error:", error);
    res.status(500).json({ error: error.message });
  }
}

// Helper function to generate a human-readable response from the product data
function formatProductDataResponse(question, productData) {
  if (!productData || productData.length === 0) {
    return "I couldn't find any relevant products matching your query.";
  }
  
  // Handle special case for empty database message
  if (productData.length === 1 && productData[0].metadata && productData[0].metadata.status === "empty_database") {
    return productData[0].pageContent || "The product database is empty. Please import CSV data first.";
  }
  
  // Extract product titles for the response
  const products = productData.map(item => {
    const metadata = item.metadata || {};
    return {
      title: metadata.title || "Unknown product",
      asin: metadata.asin || "Unknown ASIN",
      score: item.score ? (1 - item.score).toFixed(2) : "N/A" // Convert distance to similarity score
    };
  });
  
  // Create a response based on the question and found products
  let response = `Based on your question "${question}", I found these relevant products:\n\n`;
  
  products.forEach((product, index) => {
    response += `${index + 1}. ${product.title} (ASIN: ${product.asin}, Relevance: ${product.score})\n`;
  });
  
  response += "\nFor more detailed analysis on these products, you can ask specific questions about pricing trends, sales rank, or other metrics.";
  
  return response;
}