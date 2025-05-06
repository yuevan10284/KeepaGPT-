// File: pages/api/vectorsearch.js
export default async function handler(req, res) {
    try {
      // Set response headers
      res.setHeader("Content-Type", "application/json");
      
      // Make sure we have a query
      const userQuery = req.body.query || req.body.question;
      if (!userQuery) {
        return res.status(400).json({ 
          error: 'Missing search query parameter',
          status: 'error'
        });
      }
      
      // Forward the request to the backend
      const backendRes = await fetch("http://localhost:5000/api/vectorsearch", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: userQuery
        }),
      });
  
      if (!backendRes.ok) {
        // Handle error responses from backend
        const errorData = await backendRes.json().catch(() => ({ 
          error: `Backend returned status: ${backendRes.status}`
        }));
        
        throw new Error(errorData.error || `Backend error: ${backendRes.status}`);
      }
  
      // Parse the JSON response from backend
      const searchResults = await backendRes.json();
      
      // Generate a response based on the product data
      const formattedResponse = formatProductDataResponse(userQuery, searchResults.results);
      
      // Send the formatted response
      res.status(200).json({ 
        response: formattedResponse,
        rawResults: searchResults.results,
        query: userQuery,
        count: searchResults.results.length
      });
    } catch (error) {
      console.error("API route error:", error);
      res.status(500).json({ 
        error: error.message,
        status: 'error'
      });
    }
  }
  
  // Helper function to generate a human-readable response from the product data
  function formatProductDataResponse(question, productData) {
    if (!productData || productData.length === 0) {
      return "I couldn't find any relevant products matching your query.";
    }
    
    // Extract product information from the search results
    const products = productData.map(item => {
      const metadata = item.metadata || {};
      return {
        title: metadata.title || "Unknown product",
        asin: metadata.asin || "Unknown ASIN",
        // Convert score to a percentage if available
        relevance: item.score !== undefined ? 
          Math.round((1 - item.score) * 100) + '%' : 
          "N/A"
      };
    });
    
    // Create a response based on the question and found products
    let response = `Based on your query "${question}", I found ${products.length} relevant products:\n\n`;
    
    products.forEach((product, index) => {
      response += `${index + 1}. ${product.title} (ASIN: ${product.asin}, Relevance: ${product.relevance})\n`;
    });
    
    response += "\nFor more detailed analysis on these products, you can ask specific questions about pricing trends, sales rank, or other metrics.";
    
    return response;
  }