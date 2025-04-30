export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  const backendRes = await fetch('http://localhost:5000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  });
  res.setHeader('Content-Type', 'text/plain');
  backendRes.body.pipe(res);
} 