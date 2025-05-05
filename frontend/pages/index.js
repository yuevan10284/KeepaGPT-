import { useState, useRef, useEffect } from 'react';
import StarBanner from '../components/StarBanner';

const EXAMPLES = [
  'Which product has had the most stable price?',
  'Any trends suggesting I should restock?',
  "What's the best seller in this list?",
  'Which ASIN dropped in price most in the last 30 days?',
  'Summarize the price trends for each product.'
];

function ExampleMarquee({ onExample }) {
  return (
    <div className="w-full overflow-x-auto whitespace-nowrap mb-8 scrollbar-hide">
      {EXAMPLES.map((ex, i) => (
        <button
          key={i}
          className="inline-block border border-blue-400 text-blue-700 rounded px-3 py-1 bg-gray-900 hover:bg-blue-900 transition mr-2"
          onClick={() => onExample(ex)}
          type="button"
          style={{ whiteSpace: 'nowrap' }}
        >
          {ex}
        </button>
      ))}
      <style jsx global>{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamedAnswer, setStreamedAnswer] = useState('');
  const messagesEndRef = useRef(null);

  
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
  
    const userMessage = input.trim();
    setMessages([...messages, { role: 'user', content: userMessage }]);
    setIsLoading(true);
    setStreamedAnswer('');
    setInput(''); // Clear input right away for better UX
  
    try {
      // Using the API route that will handle communication with the backend
      const res = await fetch("/api/vectorsearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage }),
      });
  
      if (!res.ok) {
        throw new Error(`Error: ${res.status}`);
      }
  
      // Get the formatted text response
      const answer = await res.text();
      
      // Format the response as a chat message
      setMessages(msgs => [...msgs, { role: "assistant", content: answer }]);
    } catch (error) {
      console.error("Request error:", error);
      setMessages(msgs => [
        ...msgs,
        { role: "assistant", content: "❌ Error: " + error.message },
      ]);
    } finally {
      setIsLoading(false);
    }
  };
  

  const handleExample = (ex) => {
    setInput(ex);
  };

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-black">
      {/* Starry background effect */}
      <StarBanner numberOfStars={100} />
      <div className="w-full max-w-2xl flex flex-col items-center mt-24 z-10 relative">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-lg text-white">📊 Data powered by your Keepa CSVs</span>
          <button className="text-xs border border-blue-400 text-blue-300 rounded px-2 py-1 ml-2 bg-black hover:bg-blue-900 transition">Learn more here →</button>
        </div>
        <h1 className="text-4xl font-bold text-center mb-2 text-white">KeepaGPT</h1>
        <div className="text-center text-lg mb-6 text-gray-300">Instant insights from your Amazon product data</div>
        <form onSubmit={handleSend} className="w-full flex flex-col items-center mb-2">
          <div className="w-full flex items-center border-2 border-gray-700 rounded-lg px-4 py-3 bg-gray-900 shadow-md">
            <input
              className="flex-1 text-lg outline-none bg-transparent text-white placeholder-gray-400"
              type="text"
              placeholder="Ask about price trends, best sellers, or restock advice..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <button
              className="ml-2 text-xl text-blue-300 hover:text-blue-500 disabled:opacity-50"
              type="submit"
              disabled={isLoading || !input.trim()}
              aria-label="Send"
            >
              <span className="inline-block rotate-45">↑</span>
            </button>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <span className="flex items-center gap-1 text-gray-400 text-sm">
              <input type="checkbox" disabled className="accent-blue-500" /> Personalized <span role="img" aria-label="lock">🔒</span>
            </span>
          </div>
        </form>
        {/* Auto-scrolling example searches */}
        <ExampleMarquee onExample={handleExample} />
        {/* Chat history below search area, only show if there are messages or loading */}
        {(messages.length > 0 || isLoading) && (
          <div className="w-full max-w-2xl bg-gray-900 border border-gray-700 rounded-lg shadow p-6 flex flex-col space-y-2 min-h-[100px]">
            {messages.map((msg, i) => (
              <div key={i} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
                <span className={msg.role === 'user' ? 'bg-blue-900 text-blue-200' : 'bg-gray-800 text-gray-200'}
                  style={{ borderRadius: '0.5rem', padding: '0.5rem 1rem', display: 'inline-block', marginBottom: 2 }}>
                  {msg.content}
                </span>
              </div>
            ))}
            {isLoading && (
              <div className="text-left">
                <span className="bg-gray-800 text-gray-200" style={{ borderRadius: '0.5rem', padding: '0.5rem 1rem', display: 'inline-block' }}>
                  {streamedAnswer || <span className="animate-pulse">Thinking...</span>}
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      {/* Logo top left */}
      <div className="absolute top-4 left-4 flex items-center gap-2 z-20">
        <span className="font-bold text-xl text-pink-200">🛍️</span>
      </div>
    </div>
  );
} 