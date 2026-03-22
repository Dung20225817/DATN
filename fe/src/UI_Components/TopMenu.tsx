import { useState } from "react";
import { Link } from "react-router-dom";
import { User } from "lucide-react";
import "./TopMenu.css";
import UserSidebar from "./UserSidebar";


export default function TopMenu() {
    const [sidebarOpen, setSidebarOpen] = useState(false);
    return (
        <>
        <div className="top-menu">

            <div className="right-menu">
                <Link to="/home" className="menu-link">Trang chủ</Link>
                <Link to="/about" className="menu-link">Về chúng tôi</Link>
                <User className="user-icon" onClick={() => setSidebarOpen(true)}/>
            </div>
        </div>
        <UserSidebar
                isOpen={sidebarOpen}
                onClose={() => setSidebarOpen(false)}
            />
        </>
    );
}