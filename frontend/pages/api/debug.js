// File: pages/api/debug.js
// Quick debug endpoint to verify API routes are working

export default function handler(req, res) {
    // Return detailed information about the request and environment
    res.status(200).json({
      message: 'API route is working',
      method: req.method,
      query: req.query,
      body: req.body,
      headers: req.headers,
      timestamp: new Date().toISOString(),
      nextEnv: process.env.NODE_ENV
    });
  }