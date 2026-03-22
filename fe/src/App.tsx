import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './App.css'
import RegisterPage from "./Page_Components/RegisterPage";
import LoginPage from "./Page_Components/LoginPage";
import HomePage from "./Page_Components/HomePage";
import HandwrittenQuestionPage from "./Page_Components/HandwrittenQuestionPage";
import MultichoicePage from "./Page_Components/MultichoicePage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/multichoice" element={<MultichoicePage/>} />
        <Route path="/handwritten" element={<HandwrittenQuestionPage/>} />
      </Routes>
    </Router>
  );
}

export default App;
