import React, { useState, useEffect, useRef } from 'react';
import { documentsService } from '../services/documents';
import { queryService, QueryResponse } from '../services/query';
import Layout from '../components/Layout';
import PDFViewer from '../components/PDFViewer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<number | null>(null);
  const [highlightedChunks, setHighlightedChunks] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response: QueryResponse = await queryService.query({
        query: input,
        use_reranking: true,
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Highlight chunks in PDF viewer
      if (response.sources && response.sources.length > 0) {
        const chunkIds = response.sources.map((s) => s.chunk_id);
        setHighlightedChunks(chunkIds);
        
        // Set document if not already set
        if (response.sources[0].document_id && !selectedDocument) {
          setSelectedDocument(response.sources[0].document_id);
        }
      }
    } catch (error) {
      console.error('Error querying:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Entschuldigung, ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="flex h-[calc(100vh-8rem)]">
        <div className="flex-1 flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-8">
                Stellen Sie eine Frage, um zu beginnen
              </div>
            )}
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-3xl rounded-lg px-4 py-2 ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-gray-900 shadow'
                  }`}
                >
                  <p>{msg.content}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 text-xs opacity-75">
                      <p>Quellen:</p>
                      <ul className="list-disc list-inside">
                        {msg.sources.map((source, i) => (
                          <li key={i}>
                            Dokument {source.document_id}, Seite {source.page_number}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white rounded-lg px-4 py-2 shadow">
                  <p className="text-gray-500">Denke nach...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Stellen Sie eine Frage..."
                className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
              >
                Senden
              </button>
            </div>
          </form>
        </div>
        {selectedDocument && (
          <div className="w-1/2 border-l p-4">
            <PDFViewer
              documentId={selectedDocument}
              highlightedChunks={highlightedChunks}
            />
          </div>
        )}
      </div>
    </Layout>
  );
};

export default ChatPage;


