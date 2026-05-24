import { useEffect, useState } from "react";
import "./LoginPage.css";
import { motion } from "framer-motion";
import { Smartphone, Mail, Lock, User, Phone } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import GoogleAuthButton from "../components/GoogleAuthButton";
import { API_CONFIG } from "../config/api";
import { hasActiveAuthSession, persistAuthUser } from "../utils/authStorage";
import type { AuthUser } from "../utils/authStorage";

export default function RegisterPage() {
  const [fullname, setFullname] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (hasActiveAuthSession()) {
      navigate("/home", { replace: true });
    }
  }, [navigate]);

  const completeAuth = (user: AuthUser) => {
    persistAuthUser(user);
    navigate("/home");
  };

  const handleRegister = async () => {
    if (!fullname || !email || !phone || !password) {
      setErrorMessage("Vui lòng điền đầy đủ thông tin");
      return;
    }

    setErrorMessage("");
    setIsLoading(true);
    try {
      const res = await fetch(API_CONFIG.AUTH.REGISTER, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_name: fullname,
          email,
          phone,
          password,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Đăng ký thất bại");
      }

      completeAuth(data);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Đăng ký thất bại");
    } finally {
      setIsLoading(false);
    }
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
              <h2>Đăng ký</h2>
              <p>Tạo tài khoản mới</p>
            </div>

            <div className="inputs-container">
              <div className="input-wrapper">
                <User className="input-icon" />
                <input
                  type="text"
                  placeholder="Họ tên"
                  value={fullname}
                  onChange={(e) => setFullname(e.target.value)}
                  className="login-input"
                  disabled={isLoading}
                />
              </div>

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
                <Phone className="input-icon" />
                <input
                  type="tel"
                  placeholder="Số điện thoại"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
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
                <div
                  style={{
                    color: "#f44336",
                    fontSize: "14px",
                    padding: "10px",
                    backgroundColor: "#ffebee",
                    borderRadius: "4px",
                    marginTop: "10px",
                  }}
                >
                  {errorMessage}
                </div>
              )}
            </div>

            <button
              className="login-button"
              onClick={handleRegister}
              disabled={isLoading}
              style={{
                opacity: isLoading ? 0.7 : 1,
                cursor: isLoading ? "not-allowed" : "pointer",
              }}
            >
              {isLoading ? "Đang xử lý..." : "Đăng ký"}
            </button>

            <div className="auth-divider">
              <span>hoặc</span>
            </div>

            <GoogleAuthButton
              disabled={isLoading}
              text="signup_with"
              onSuccess={completeAuth}
              onError={setErrorMessage}
            />

            <p className="login-footer">
              <Link to="/login" className="register-link">
                Đăng nhập
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
