const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const CSV_DIR = path.join(__dirname, '../csv');
let keepaData = {};

// Load and parse all CSVs on startup
function loadCSVs() {
  keepaData = {};
  if (!fs.existsSync(CSV_DIR)) return;
  const files = fs.readdirSync(CSV_DIR).filter(f => f.endsWith('.csv'));
  for (const file of files) {
    const content = fs.readFileSync(path.join(CSV_DIR, file), 'utf8');
    try {
      const records = parse(content, { columns: true });
      keepaData[file] = records;
    } catch (e) {
      console.error(`Failed to parse ${file}:`, e);
    }
  }
}

loadCSVs();

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

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
}); 