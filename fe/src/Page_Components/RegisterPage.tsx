import { useState } from "react";
import "../UI_Components/LoginPage.css";
import { motion } from "framer-motion";
import { Smartphone, Mail, Lock, User, Phone } from "lucide-react";
import { Link } from "react-router-dom";

export default function RegisterPage() {
    const [fullname, setFullname] = useState("");
    const [email, setEmail] = useState("");
    const [phone, setPhone] = useState("");
    const [password, setPassword] = useState("");

    const handleRegister = () => {
        console.log("Register with:", fullname, email, phone, password);
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
                        {/* Header */}
                        <div className="login-header">
                            <Smartphone className="login-icon" />
                            <h2>Đăng ký</h2>
                            <p>Tạo tài khoản mới</p>
                        </div>

                        {/* Inputs */}
                        <div className="inputs-container">
                            {/* Họ tên */}
                            <div className="input-wrapper">
                                <User className="input-icon" />
                                <input
                                    type="text"
                                    placeholder="Họ tên"
                                    value={fullname}
                                    onChange={(e) => setFullname(e.target.value)}
                                    className="login-input"
                                />
                            </div>

                            {/* Email */}
                            <div className="input-wrapper">
                                <Mail className="input-icon" />
                                <input
                                    type="email"
                                    placeholder="Email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className="login-input"
                                />
                            </div>

                            {/* Số điện thoại */}
                            <div className="input-wrapper">
                                <Phone className="input-icon" />
                                <input
                                    type="tel"
                                    placeholder="Số điện thoại"
                                    value={phone}
                                    onChange={(e) => setPhone(e.target.value)}
                                    className="login-input"
                                />
                            </div>

                            {/* Mật khẩu */}
                            <div className="input-wrapper">
                                <Lock className="input-icon" />
                                <input
                                    type="password"
                                    placeholder="Mật khẩu"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="login-input"
                                />
                            </div>

                        </div>

                        {/* Button */}
                        <button className="login-button" onClick={handleRegister}>
                            Đăng ký
                        </button>

                        <p className="login-footer">
                            <Link to="/" className="register-link">
                                Đăng nhập
                            </Link>
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
