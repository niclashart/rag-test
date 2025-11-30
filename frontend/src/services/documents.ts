import api from './api';

export interface Document {
  id: number;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  created_at: string;
  metadata: Record<string, any>;
}

export interface Chunk {
  id: string;
  page_number: number;
  text: string;
  chunk_index: number;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export const documentsService = {
  async upload(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  async list(): Promise<Document[]> {
    const response = await api.get('/api/documents');
    return response.data;
  },

  async get(id: number): Promise<Document> {
    const response = await api.get(`/api/documents/${id}`);
    return response.data;
  },

  async delete(id: number): Promise<void> {
    await api.delete(`/api/documents/${id}`);
  },

  async ingest(id: number): Promise<any> {
    const response = await api.post(`/api/documents/${id}/ingest`);
    return response.data;
  },

  getPreviewUrl(id: number): string {
    return `${api.defaults.baseURL}/api/documents/${id}/preview`;
  },

  async getChunks(id: number): Promise<Chunk[]> {
    const response = await api.get(`/api/documents/${id}/chunks`);
    return response.data;
  },

  async getChunk(documentId: number, chunkId: string): Promise<Chunk> {
    const response = await api.get(`/api/documents/${documentId}/chunks/${chunkId}`);
    return response.data;
  },
};

