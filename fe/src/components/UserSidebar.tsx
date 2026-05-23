import { useState, useEffect } from "react";
import { X } from "lucide-react";
import "./UserSidebar.css";

interface UserSidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

interface User {
    user_name: string;
    email: string;
    uid?: number;
    [key: string]: unknown;
}

export default function UserSidebar({ isOpen, onClose }: UserSidebarProps) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const userStr = localStorage.getItem("user");
        if (userStr) {
            try {
                const userObj = JSON.parse(userStr);
                setUser(userObj);
            } catch (err) {
                console.error("Error parsing user from localStorage:", err);
            }
        }
    }, []);

    const handleSignOut = async () => {
        setIsLoading(true);
        try {
            // No backend session endpoint is active; client-side logout is sufficient.
        } catch (err) {
            console.error("Logout error:", err);
        } finally {
            // Clear localStorage regardless of API success
            localStorage.removeItem("token");
            localStorage.removeItem("uid");
            localStorage.removeItem("user_name");
            localStorage.removeItem("email");
            localStorage.removeItem("phone");
            localStorage.removeItem("user");
            
            setIsLoading(false);
            window.location.href = "/";
        }
    };

    return (
        <div className={`sidebar-overlay ${isOpen ? "active" : ""}`} onClick={onClose}>
            <div className="sidebar" onClick={(e) => e.stopPropagation()}>
                <div className="sidebar-header">
                    <h3>Thông tin người dùng</h3>
                    <button className="close-btn" onClick={onClose}>
                        <X size={20} />
                    </button>
                </div>
                <div className="sidebar-content">
                    {user ? (
                        <>
                            <p><strong>Tên:</strong> {user.user_name}</p>
                            <p><strong>Email:</strong> {user.email}</p>
                        </>
                    ) : (
                        <p>Không có thông tin người dùng</p>
                    )}

                    <button 
                        className="signout-btn" 
                        onClick={handleSignOut}
                        disabled={isLoading}
                        style={{
                            opacity: isLoading ? 0.7 : 1,
                            cursor: isLoading ? "not-allowed" : "pointer"
                        }}
                    >
                        {isLoading ? "Đang đăng xuất..." : "Đăng xuất"}
                    </button>
                </div>
            </div>
        </div>
    );
}
