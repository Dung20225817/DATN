import { useState } from "react";
import "../UI_Components/LoginPage.css";
import { motion } from "framer-motion";
import { Smartphone, Mail, Lock } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { API_CONFIG } from "../config/api";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const loginRequest = async (email: string, password: string) => {
    try {
      setErrorMessage("");
      setIsLoading(true);
      
      const res = await fetch(API_CONFIG.AUTH.LOGIN, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Login failed");
      }

      return await res.json();
    } catch (err) {
      console.error("Login error:", err);
      setErrorMessage(err instanceof Error ? err.message : "Login failed");
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!email || !password) {
      setErrorMessage("Vui lòng nhập email và mật khẩu");
      return;
    }

    const user = await loginRequest(email, password);

    if (!user) {
      return;
    }

    // Save user info to localStorage
    localStorage.setItem("token", user.token || "");
    localStorage.setItem("uid", user.uid?.toString() || "");
    localStorage.setItem("user_name", user.user_name || "");
    localStorage.setItem("email", user.email || "");
    localStorage.setItem("phone", user.phone || "");
    localStorage.setItem("user", JSON.stringify(user));

    navigate("/home");
  };



  return (
    <div className="login-page-container">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="login-card-wrapper"
      >
        <div className="login-card">
          <div className="login-card-content">
            <div className="login-header">
              <Smartphone className="login-icon" />
              <h2>Đăng nhập</h2>
              <p>Chấm điểm bằng hình ảnh</p>
            </div>

            <div className="inputs-container">
              <div className="input-wrapper">
                <Mail className="input-icon" />
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="login-input"
                  disabled={isLoading}
                />
              </div>

              <div className="input-wrapper">
                <Lock className="input-icon" />
                <input
                  type="password"
                  placeholder="Mật khẩu"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="login-input"
                  disabled={isLoading}
                />
              </div>

              {errorMessage && (
                <div style={{
                  color: "#f44336",
                  fontSize: "14px",
                  padding: "10px",
                  backgroundColor: "#ffebee",
                  borderRadius: "4px",
                  marginTop: "10px"
                }}>
                  {errorMessage}
                </div>
              )}
            </div>

            <button 
              className="login-button" 
              onClick={handleLogin}
              disabled={isLoading}
              style={{
                opacity: isLoading ? 0.7 : 1,
                cursor: isLoading ? "not-allowed" : "pointer"
              }}
            >
              {isLoading ? "Đang xử lý..." : "Đăng nhập"}
            </button>

            <p className="login-footer">
              Chưa có tài khoản?{" "}
              <Link to="/register" className="register-link">
                Đăng ký
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
