import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { documentsService } from '../services/documents';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import 'react-pdf/dist/esm/Page/Page.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url
).toString();

interface PDFViewerProps {
  documentId: number;
  highlightedChunks?: string[];
}

const PDFViewer: React.FC<PDFViewerProps> = ({ documentId, highlightedChunks = [] }) => {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [chunks, setChunks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadChunks();
  }, [documentId]);

  const loadChunks = async () => {
    try {
      const documentChunks = await documentsService.getChunks(documentId);
      setChunks(documentChunks);
    } catch (error) {
      console.error('Error loading chunks:', error);
    } finally {
      setLoading(false);
    }
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const onDocumentLoadError = (error: Error) => {
    console.error('Error loading PDF:', error);
    setLoading(false);
  };

  const previewUrl = documentsService.getPreviewUrl(documentId);

  // Get chunks for current page
  const pageChunks = chunks.filter((c) => c.page_number === pageNumber);
  const highlightedPageChunks = pageChunks.filter((c) =>
    highlightedChunks.includes(c.id)
  );

  return (
    <div className="h-full flex flex-col">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-medium">PDF-Vorschau</h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            disabled={pageNumber <= 1}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Zurück
          </button>
          <span className="text-sm">
            Seite {pageNumber} von {numPages || '?'}
          </span>
          <button
            onClick={() => setPageNumber((p) => Math.min(numPages || 1, p + 1))}
            disabled={pageNumber >= (numPages || 1)}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Weiter
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto border bg-gray-100">
        {loading ? (
          <div className="flex items-center justify-center h-full">Laden...</div>
        ) : (
          <div className="relative">
            <Document
              file={previewUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div>PDF wird geladen...</div>}
            >
              <Page
                pageNumber={pageNumber}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="mx-auto"
                width={undefined}
              />
            </Document>
            {highlightedPageChunks.length > 0 && (
              <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                {highlightedPageChunks.map((chunk) => {
                  const bbox = chunk.bbox;
                  if (!bbox) return null;
                  return (
                    <div
                      key={chunk.id}
                      className="absolute bg-yellow-300 bg-opacity-50 border-2 border-yellow-500"
                      style={{
                        left: `${bbox.x}px`,
                        top: `${bbox.y}px`,
                        width: `${bbox.width}px`,
                        height: `${bbox.height}px`,
                      }}
                    />
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {highlightedPageChunks.length > 0 && (
        <div className="mt-4 p-2 bg-yellow-50 border border-yellow-200 rounded">
          <p className="text-sm font-medium text-yellow-800">
            {highlightedPageChunks.length} relevante Chunk(s) auf dieser Seite
          </p>
        </div>
      )}
    </div>
  );
};

export default PDFViewer;


