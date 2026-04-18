/**
 * API Configuration
 * Centralized API endpoint management
 */

const envApiBaseUrl = (import.meta.env.VITE_API_URL || "").trim();
const isLoopbackApiBase = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(envApiBaseUrl);

const API_BASE_URL = import.meta.env.DEV
  // In dev, force same-origin (/api) when env points to localhost so LAN devices work via Vite proxy.
  ? (envApiBaseUrl && !isLoopbackApiBase ? envApiBaseUrl : "")
  // In production behind reverse proxy (Nginx/Caddy), default to same-origin.
  : (envApiBaseUrl || "");

export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  
  // Auth endpoints
  AUTH: {
    LOGIN: `${API_BASE_URL}/api/login`,
    REGISTER: `${API_BASE_URL}/api/register`,
  },
  
  // Handwritten essay endpoints (NEW - 2 step workflow)
  HANDWRITTEN: {
    UPLOAD_ANSWER_KEY: `${API_BASE_URL}/api/handwritten/upload-answer-key`,
    SAVE_ANSWER_KEY: `${API_BASE_URL}/api/handwritten/save-answer-key`,
    UPLOAD: `${API_BASE_URL}/api/handwritten/upload`,
    LIST_ANSWER_KEYS: (uid: number) => `${API_BASE_URL}/api/handwritten/answer-keys/${uid}`,
    GET_ANSWER_KEY: (id: number, uid: number) => `${API_BASE_URL}/api/handwritten/answer-key/${id}?uid=${uid}`,
    DOWNLOAD_ANSWER_KEY: (id: number, uid: number) => `${API_BASE_URL}/api/handwritten/answer-key/${id}/download?uid=${uid}`,
    DELETE_ANSWER_KEY: (id: number, uid: number) => `${API_BASE_URL}/api/handwritten/answer-key/${id}?uid=${uid}`,
  },
  
  // OMR grading endpoints
  OMR: {
    GRADE: `${API_BASE_URL}/api/omr/grade`,
    GRADE_BATCH: `${API_BASE_URL}/api/omr/grade-batch`,
    LIST_FORM_SAMPLES: `${API_BASE_URL}/api/omr/form-samples`,
    LIST_FORM_PROFILES: `${API_BASE_URL}/api/omr/form-profiles`,
    GET_FORM_PROFILE: (code: string) => `${API_BASE_URL}/api/omr/form-profiles/${code}`,
    SAVE_FORM_PROFILE: `${API_BASE_URL}/api/omr/form-profiles`,
    SUGGEST_CROP: `${API_BASE_URL}/api/omr/suggest-crop`,
    CREATE_ASSIGNMENT: `${API_BASE_URL}/api/omr/assignments`,
    LIST_ASSIGNMENTS: (uid: number) => `${API_BASE_URL}/api/omr/assignments/${uid}`,
    UPDATE_ASSIGNMENT: (uid: number, aid: number) => `${API_BASE_URL}/api/omr/assignments/${uid}/${aid}`,
    DELETE_ASSIGNMENT: (uid: number, aid: number) => `${API_BASE_URL}/api/omr/assignments/${uid}/${aid}`,
  },
  
};

/**
 * Common fetch wrapper with error handling
 */
export async function apiCall<T>(
  url: string,
  options?: RequestInit
): Promise<{ data: T; error: null } | { data: null; error: string }> {
  try {
    const headers = new Headers(options?.headers);
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        detail: "Unknown error",
      }));
      return {
        data: null,
        error: errorData.detail || errorData.message || "API Error",
      };
    }

    const data = await response.json();
    return { data, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Network error",
    };
  }
}

export default API_CONFIG;
