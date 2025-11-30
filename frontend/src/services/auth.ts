import api from './api';

export interface LoginCredentials {
  username: string; // email
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
}

export interface User {
  id: number;
  email: string;
  is_active: boolean;
}

export const authService = {
  async login(credentials: LoginCredentials) {
    const formData = new FormData();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    const response = await api.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    if (response.data.access_token) {
      localStorage.setItem('access_token', response.data.access_token);
    }
    
    return response.data;
  },

  async register(data: RegisterData) {
    const response = await api.post('/api/auth/register', data);
    return response.data;
  },

  async getCurrentUser(): Promise<User> {
    const response = await api.get('/api/auth/me');
    return response.data;
  },

  logout() {
    localStorage.removeItem('access_token');
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  },
};


