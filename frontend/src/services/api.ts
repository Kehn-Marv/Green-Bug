import axios from 'axios';

// Use a relative base so Vite dev server proxy (configured in vite.config.ts)
// forwards requests to the backend during development.
const API_BASE_URL = import.meta.env.PROD ? '/api' : '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds default timeout for comprehensive analysis
});

// Health endpoints
export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getDetailedHealth = async () => {
  const response = await api.get('/health/detailed');
  return response.data;
};

// Analysis endpoints
export const analyzeImage = async (file: File, options: any = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add analysis options as query parameters
  const params = new URLSearchParams();
  if (options.stripExif !== undefined) params.append('strip_exif', options.stripExif.toString());
  if (options.enableLearning !== undefined) params.append('enable_learning', options.enableLearning.toString());
  if (options.generateReport !== undefined) params.append('generate_report', options.generateReport.toString());
  if (options.targetLayer) params.append('target_layer', options.targetLayer);
  
  const queryString = params.toString();
  const url = queryString ? `/analyze?${queryString}` : '/analyze';

  // Include query params in the request URL so backend receives options
  // analysis can be long running; allow override timeout via options.timeout (ms)
  const requestTimeout = typeof options.timeout === 'number' ? options.timeout : 120000; // default 120s

  const response = await api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: requestTimeout,
  });

  return response.data;
};

export const analyzeBatch = async (files: File[]) => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  
  const response = await api.post('/analyze/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Job-based analysis (submit & poll)
export const submitAnalyze = async (file: File, options: any = {}) => {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (options.stripExif !== undefined) params.append('strip_exif', options.stripExif.toString());
  if (options.enableLearning !== undefined) params.append('enable_learning', options.enableLearning.toString());
  if (options.generateReport !== undefined) params.append('generate_report', options.generateReport.toString());
  if (options.targetLayer) params.append('target_layer', options.targetLayer);

  const queryString = params.toString();
  const url = queryString ? `/analyze/submit?${queryString}` : '/analyze/submit';

  const response = await api.post(url, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 10000 // quick acknowledgement
  });

  return response.data as { job_id: string; estimated_seconds?: number };
};

export const getAnalyzeResult = async (jobId: string) => {
  const response = await api.get(`/analyze/result/${jobId}`);
  return response.data as { job_id: string; status: string; result?: any; error?: any };
};

// SSE helper to open EventSource for job progress
export const openAnalyzeEvents = (jobId: string, onMessage: (data: any) => void, onOpen?: () => void, onClose?: () => void) => {
  const baseUrl = api.defaults.baseURL || '';
  const url = `${baseUrl}/analyze/events/${jobId}`;

  let es: EventSource | null = null;
  let closed = false;
  let attempt = 0;

  const connect = () => {
    if (closed) return;
    
    es = new EventSource(url);
    
    es.onopen = () => {
      attempt = 0;
      if (onOpen) onOpen();
    };
    
    es.onmessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        onMessage(data);
      } catch (err) {
        console.warn('Failed to parse SSE message:', err);
      }
    };
    
    es.onerror = () => {
      if (es) {
        try { es.close(); } catch (e) {}
      }
      es = null;
      
      if (closed) return;
      
      // Exponential backoff for reconnection
      attempt += 1;
      if (attempt <= 5) {
        const backoff = Math.min(10000, 1000 * Math.pow(2, attempt));
        setTimeout(() => connect(), backoff);
      } else {
        console.error('SSE connection failed after 5 attempts');
        if (onClose) onClose();
      }
    };
  };

  connect();

  return {
    close: () => {
      closed = true;
      if (es) {
        try { es.close(); } catch (e) {}
      }
      if (onClose) onClose();
    }
  };
};

// Self-learning endpoints
export const processConsent = async (candidateId: string, userId: string, consentGiven: boolean, humanLabel?: string) => {
  const response = await api.post('/analyze/consent', {
    candidate_id: candidateId,
    user_id: userId,
    consent_given: consentGiven,
    human_label: humanLabel
  });
  return response.data;
};

export const getLearningStats = async () => {
  const response = await api.get('/analyze/learning/stats');
  return response.data;
};

export const cleanupLearningData = async () => {
  const response = await api.post('/analyze/learning/cleanup');
  return response.data;
};

// Stats endpoints
export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const getFamilies = async () => {
  const response = await api.get('/families');
  return response.data;
};

// Error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Unable to connect to the server. Please check if the API is running.');
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred');
    }
  }
);

export default api;