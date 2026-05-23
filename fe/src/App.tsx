import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './App.css'
import LandingPage from "./pages/LandingPage";
import RegisterPage from "./pages/RegisterPage";
import LoginPage from "./pages/LoginPage";
import HomePage from "./pages/HomePage";
import MultichoicePage from "./features/omr/pages/MultichoicePage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/multichoice" element={<MultichoicePage/>} />
        <Route path="/multichoice/record-detail/:testId/:recordId?" element={<MultichoicePage/>} />
      </Routes>
    </Router>
  );
}

export default App;
