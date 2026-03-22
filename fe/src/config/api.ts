/**
 * API Configuration
 * Centralized API endpoint management
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
    CREATE_TEMPLATE: `${API_BASE_URL}/api/omr/template`,
    SAVE_TEMPLATE: `${API_BASE_URL}/api/omr/template/save`,
    SUGGEST_CROP: `${API_BASE_URL}/api/omr/suggest-crop`,
    LIST_TESTS: (uid: number) => `${API_BASE_URL}/api/omr/tests/${uid}`,
    DELETE_TEST: (uid: number, omrid: number) => `${API_BASE_URL}/api/omr/tests/${uid}/${omrid}`,
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
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": options?.headers instanceof Headers 
          ? undefined 
          : "application/json",
        ...options?.headers,
      },
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
