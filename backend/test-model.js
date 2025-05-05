import { pipeline } from '@xenova/transformers';

async function testModel() {
  try {
    console.log('Starting model download...');
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Model loaded successfully');
    const embedding = await embedder('Test sentence', { pooling: 'mean', normalize: true });
    console.log('Embedding generated:', embedding.data.slice(0, 5));
  } catch (error) {
    console.error('Model load error:', error.message);
    console.error('Full error:', error);
  }
}

testModel();