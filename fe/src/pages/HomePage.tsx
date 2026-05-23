import TopMenu from "../components/TopMenu";
import { Link } from "react-router-dom";
import "./HomePage.css";
import { motion } from "framer-motion";

export default function HomePage() {
  const featureItems = [
    {
      title: "Quét camera",
      description: "Tự nhận diện marker góc và chụp phiếu khi khung đã ổn định.",
    },
    {
      title: "Chấm theo lô",
      description: "Tải nhiều ảnh cùng lúc để xử lý nhanh cả một tập bài thi.",
    },
    {
      title: "Nhiều mã đề",
      description: "Quản lý các bộ đáp án khác nhau trong cùng một bài kiểm tra.",
    },
    {
      title: "Xuất kết quả",
      description: "Tổng hợp điểm và xuất Excel hoặc PDF trực tiếp từ trình duyệt.",
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 },
    },
  };

  return (
    <div className="home-page-wrap">
      <TopMenu />

      {/* Hero */}
      <motion.div
        className="images-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Link to="/multichoice" className="image-wrapper">
          <img src="/image/multichoice.jpg" alt="Multiple Choice" className="home-image" />
          <div className="image-text-wrap">
            <div className="image-text">Chấm Trắc Nghiệm</div>
            <div className="image-subtext">Chấm điểm bài trắc nghiệm tự động bằng camera</div>
            <div className="image-cta">Bắt đầu ngay →</div>
          </div>
        </Link>
      </motion.div>

      {/* Features */}
      <div className="features-section">
        <h2 className="features-heading">Tính năng nổi bật</h2>
        <motion.div
          className="features-container"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {featureItems.map((item, index) => (
            <motion.div
              className="feature-box"
              key={index}
              variants={itemVariants}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="feature-index">{String(index + 1).padStart(2, "0")}</div>
              <div className="feature-box-content">
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
