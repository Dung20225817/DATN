import TopMenu from "../UI_Components/TopMenu";
import { Link } from "react-router-dom";
import "../UI_Components/HomePage.css";
import { motion } from "framer-motion";

export default function HomePage() {
  const featureItems = [
    {
      img: "/image/feature1.jpg",
      title: "Feature 1",
      description: "Mô tả chức năng 1",
    },
    {
      img: "/image/feature2.jpg",
      title: "Feature 2",
      description: "Mô tả chức năng 2",
    },
    {
      img: "/image/feature3.jpg",
      title: "Feature 3",
      description: "Mô tả chức năng 3",
    },
    {
      img: "/image/feature4.jpg",
      title: "Feature 4",
      description: "Mô tả chức năng 4",
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
    <div>
      <TopMenu />
      {/* nội dung trang */}
      <motion.div 
        className="images-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <Link to="/multichoice" className="image-wrapper">
          <img src="/image/multichoice.jpg" alt="Multiple Choice" className="home-image" />
          <div className="image-text">Chấm Trắc Nghiệm</div>
        </Link>
        <Link to="/handwritten" className="image-wrapper">
          <img src="/image/handwritten.jpg" alt="Handwritten Question" className="home-image" />
          <div className="image-text">Chấm Essay Viết Tay</div>
        </Link>
      </motion.div>

      {/* 4 ô chữ nhật text */}
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
            <img src={item.img} alt={item.title} className="feature-img" />
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
