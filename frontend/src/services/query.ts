import api from './api';

export interface QueryRequest {
  query: string;
  use_reranking?: boolean;
}

export interface SourceInfo {
  chunk_id: string;
  document_id?: number;
  page_number?: number;
  similarity?: number;
}

export interface QueryResponse {
  answer: string;
  query: string;
  sources: SourceInfo[];
  retrieval_time: number;
  generation_time: number;
}

export interface QueryHistoryItem {
  id: number;
  query: string;
  answer?: string;
  sources: string[];
  created_at: string;
}

export const queryService = {
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await api.post('/api/query', request);
    return response.data;
  },

  async getHistory(limit: number = 50): Promise<QueryHistoryItem[]> {
    const response = await api.get(`/api/query/history?limit=${limit}`);
    return response.data;
  },
};


