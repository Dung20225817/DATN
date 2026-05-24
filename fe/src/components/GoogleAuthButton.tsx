import { useEffect, useRef, useState } from "react";
import { API_CONFIG, GOOGLE_CLIENT_ID } from "../config/api";
import type { AuthUser } from "../utils/authStorage";

const GOOGLE_SCRIPT_SRC = "https://accounts.google.com/gsi/client";

type GoogleButtonStatus = "idle" | "loading" | "ready" | "error";

interface GoogleCredentialResponse {
  credential?: string;
  select_by?: string;
}

interface GoogleAccountsId {
  initialize: (options: {
    client_id: string;
    callback: (response: GoogleCredentialResponse) => void;
  }) => void;
  renderButton: (
    parent: HTMLElement,
    options: {
      theme?: "outline" | "filled_blue" | "filled_black";
      size?: "large" | "medium" | "small";
      shape?: "rectangular" | "pill" | "circle" | "square";
      width?: string;
      text?: "signin_with" | "signup_with" | "continue_with" | "signin";
      locale?: string;
    },
  ) => void;
}

declare global {
  interface Window {
    google?: {
      accounts: {
        id: GoogleAccountsId;
      };
    };
  }
}

interface GoogleAuthButtonProps {
  disabled?: boolean;
  onSuccess: (user: AuthUser) => void;
  onError: (message: string) => void;
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
}

let googleScriptPromise: Promise<void> | null = null;

function loadGoogleScript() {
  if (window.google?.accounts?.id) {
    return Promise.resolve();
  }

  if (!googleScriptPromise) {
    googleScriptPromise = new Promise<void>((resolve, reject) => {
      const existingScript = document.querySelector<HTMLScriptElement>(
        `script[src="${GOOGLE_SCRIPT_SRC}"]`,
      );

      if (existingScript) {
        existingScript.addEventListener("load", () => resolve(), { once: true });
        existingScript.addEventListener("error", () => reject(new Error("Google script failed")), {
          once: true,
        });
        return;
      }

      const script = document.createElement("script");
      script.src = GOOGLE_SCRIPT_SRC;
      script.async = true;
      script.defer = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Google script failed"));
      document.head.appendChild(script);
    });
  }

  return googleScriptPromise;
}

async function authenticateWithGoogle(credential: string): Promise<AuthUser> {
  const res = await fetch(API_CONFIG.AUTH.GOOGLE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ credential }),
  });

  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(data?.detail || "Google login failed");
  }

  return data;
}

export default function GoogleAuthButton({
  disabled = false,
  onSuccess,
  onError,
  text = "continue_with",
}: GoogleAuthButtonProps) {
  const buttonRef = useRef<HTMLDivElement | null>(null);
  const onSuccessRef = useRef(onSuccess);
  const onErrorRef = useRef(onError);
  const disabledRef = useRef(disabled);
  const [status, setStatus] = useState<GoogleButtonStatus>("idle");

  useEffect(() => {
    onSuccessRef.current = onSuccess;
    onErrorRef.current = onError;
    disabledRef.current = disabled;
  }, [disabled, onError, onSuccess]);

  useEffect(() => {
    let cancelled = false;

    const renderGoogleButton = async () => {
      if (!GOOGLE_CLIENT_ID) {
        setStatus("error");
        return;
      }

      setStatus("loading");
      try {
        await loadGoogleScript();
        if (cancelled || !buttonRef.current || !window.google?.accounts?.id) {
          return;
        }

        buttonRef.current.innerHTML = "";
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: async (response) => {
            if (disabledRef.current) {
              return;
            }

            if (!response.credential) {
              onErrorRef.current("Google login failed");
              return;
            }

            try {
              const user = await authenticateWithGoogle(response.credential);
              onSuccessRef.current(user);
            } catch (error) {
              onErrorRef.current(error instanceof Error ? error.message : "Google login failed");
            }
          },
        });

        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: "outline",
          size: "large",
          shape: "rectangular",
          width: "340",
          text,
          locale: "vi",
        });
        setStatus("ready");
      } catch (error) {
        if (!cancelled) {
          setStatus("error");
          onErrorRef.current(error instanceof Error ? error.message : "Google login failed");
        }
      }
    };

    void renderGoogleButton();

    return () => {
      cancelled = true;
      if (buttonRef.current) {
        buttonRef.current.innerHTML = "";
      }
    };
  }, [text]);

  if (!GOOGLE_CLIENT_ID) {
    return (
      <button className="google-auth-fallback" type="button" disabled>
        Google login is not configured
      </button>
    );
  }

  return (
    <div className="google-auth-container" aria-disabled={disabled}>
      <div ref={buttonRef} className={disabled ? "google-auth-button disabled" : "google-auth-button"} />
      {status === "loading" && <div className="google-auth-status">Loading Google...</div>}
      {status === "error" && <div className="google-auth-status error">Google login unavailable</div>}
      {status === "ready" && disabled && <div className="google-auth-disabled-overlay" />}
    </div>
  );
}
