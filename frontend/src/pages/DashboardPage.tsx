import React, { useState, useEffect } from 'react';
import { documentsService, Document } from '../services/documents';
import Layout from '../components/Layout';

const DashboardPage: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [ingesting, setIngesting] = useState<number | null>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await documentsService.list();
      setDocuments(docs);
    } catch (error) {
      console.error('Error loading documents:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      await documentsService.upload(file);
      await loadDocuments();
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Fehler beim Hochladen der Datei');
    } finally {
      setUploading(false);
    }
  };

  const handleIngest = async (documentId: number) => {
    setIngesting(documentId);
    try {
      await documentsService.ingest(documentId);
      await loadDocuments();
    } catch (error) {
      console.error('Error ingesting document:', error);
      alert('Fehler beim Indizieren des Dokuments');
    } finally {
      setIngesting(null);
    }
  };

  const handleDelete = async (documentId: number) => {
    if (!confirm('Möchten Sie dieses Dokument wirklich löschen?')) return;

    try {
      await documentsService.delete(documentId);
      await loadDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('Fehler beim Löschen des Dokuments');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Layout>
      <div className="px-4 py-6 sm:px-0">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Dokumente</h2>
          <p className="mt-1 text-sm text-gray-500">
            Laden Sie Dokumente hoch und indizieren Sie sie für die RAG-Pipeline
          </p>
        </div>

        <div className="mb-6">
          <label className="block">
            <span className="sr-only">Datei auswählen</span>
            <input
              type="file"
              onChange={handleFileUpload}
              disabled={uploading}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
              accept=".pdf,.docx,.txt,.md,.csv,.xlsx,.html,.json"
            />
          </label>
          {uploading && <p className="mt-2 text-sm text-gray-500">Hochladen...</p>}
        </div>

        {loading ? (
          <div className="text-center py-12">Laden...</div>
        ) : documents.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            Noch keine Dokumente hochgeladen
          </div>
        ) : (
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <ul className="divide-y divide-gray-200">
              {documents.map((doc) => (
                <li key={doc.id}>
                  <div className="px-4 py-4 sm:px-6">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                            {doc.file_type.toUpperCase()}
                          </span>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {doc.filename}
                          </div>
                          <div className="text-sm text-gray-500">
                            {formatFileSize(doc.file_size)} • {doc.status}
                          </div>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        {doc.status !== 'indexed' && (
                          <button
                            onClick={() => handleIngest(doc.id)}
                            disabled={ingesting === doc.id}
                            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
                          >
                            {ingesting === doc.id ? 'Indizieren...' : 'Indizieren'}
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                        >
                          Löschen
                        </button>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default DashboardPage;


