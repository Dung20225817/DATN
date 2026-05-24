export interface AuthUser {
  uid?: number | string;
  email?: string;
  user_name?: string;
  phone?: string;
  token?: string;
}

const AUTH_KEYS = ["token", "uid", "user_name", "email", "phone", "user"];

export function persistAuthUser(user: AuthUser) {
  clearLegacyAuthStorage();
  sessionStorage.setItem("token", user.token || "");
  sessionStorage.setItem("uid", user.uid?.toString() || "");
  sessionStorage.setItem("user_name", user.user_name || "");
  sessionStorage.setItem("email", user.email || "");
  sessionStorage.setItem("phone", user.phone || "");
  sessionStorage.setItem("user", JSON.stringify(user));
}

export function clearLegacyAuthStorage() {
  AUTH_KEYS.forEach((key) => localStorage.removeItem(key));
}

export function clearAuthSession() {
  AUTH_KEYS.forEach((key) => {
    sessionStorage.removeItem(key);
    localStorage.removeItem(key);
  });
}

export function getAuthUser(): AuthUser | null {
  const userRaw = sessionStorage.getItem("user");
  if (!userRaw) {
    return null;
  }

  try {
    return JSON.parse(userRaw) as AuthUser;
  } catch {
    return null;
  }
}

export function getAuthUid() {
  const uid = (sessionStorage.getItem("uid") || "").trim();
  if (uid && uid !== "0") {
    return Number(uid);
  }

  const user = getAuthUser();
  const userUid = user?.uid != null ? String(user.uid).trim() : "";
  if (!userUid || userUid === "0") {
    return 0;
  }
  return Number(userUid) || 0;
}

export function hasActiveAuthSession() {
  const token = (sessionStorage.getItem("token") || "").trim();
  const uid = (sessionStorage.getItem("uid") || "").trim();
  if (token && uid && uid !== "0") {
    return true;
  }

  const user = getAuthUser();
  return Boolean(user?.uid && String(user.uid).trim() && String(user.uid).trim() !== "0");
}
