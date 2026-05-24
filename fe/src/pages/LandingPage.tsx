import { Link } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Lock,
  ScanLine,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { motion } from "framer-motion";
import { hasActiveAuthSession } from "../utils/authStorage";
import "./LandingPage.css";

const benefitItems = [
  {
    icon: ScanLine,
    title: "Chấm phiếu nhanh",
    description:
      "Nhận diện vùng tô đáp án, mã đề và thông tin bài làm để giảm thao tác nhập điểm thủ công.",
  },
  {
    icon: CheckCircle2,
    title: "Kết quả nhất quán",
    description:
      "So khớp theo đáp án đã cấu hình, hạn chế sai lệch khi phải xử lý nhiều lớp hoặc nhiều mã đề.",
  },
  {
    icon: BarChart3,
    title: "Theo dõi sau chấm",
    description:
      "Lưu bản ghi, xem thống kê và xuất dữ liệu để giáo viên tiếp tục tổng hợp điểm.",
  },
];

const featureItems = [
  {
    icon: Camera,
    title: "Smart Camera Scanner",
    description: "Khóa khung khi đủ marker, đọc frame trực tiếp trên trình duyệt và gửi ảnh ổn định hơn.",
  },
  {
    icon: UploadCloud,
    title: "Chấm theo lô",
    description: "Tải nhiều ảnh bài thi cùng lúc, xử lý tuần tự và theo dõi trạng thái từng bản ghi.",
  },
  {
    icon: FileSpreadsheet,
    title: "Quản lý đáp án",
    description: "Tạo bài thi, cấu hình số câu, mã đề, điểm từng câu và đáp án cho từng phiên bản đề.",
  },
  {
    icon: Download,
    title: "Xuất kết quả",
    description: "Tải bảng điểm và dữ liệu chấm để dùng tiếp trong báo cáo hoặc hồ sơ lớp học.",
  },
];

const workflowItems = [
  "Tạo bài thi và cấu hình form trả lời.",
  "Nhập đáp án cho từng mã đề.",
  "Quét bằng camera hoặc tải ảnh phiếu lên.",
  "Kiểm tra bản ghi, thống kê và xuất file.",
];

export default function LandingPage() {
  const signedInTarget = hasActiveAuthSession() ? "/home" : null;

  return (
    <div className="landing-page">
      <header className="landing-header" aria-label="Điều hướng VeritaAI">
        <Link to="/" className="landing-brand" aria-label="VeritaAI">
          <span className="landing-brand-mark">V</span>
          <span>VeritaAI</span>
        </Link>
        <nav className="landing-nav" aria-label="Liên kết trang">
          <a href="#benefits">Lợi ích</a>
          <a href="#features">Tính năng</a>
          <a href="#workflow">Quy trình</a>
          <a href="#privacy">Bảo mật</a>
        </nav>
        <div className="landing-actions">
          <Link to={signedInTarget || "/login"} className="landing-link-button">
            Đăng nhập
          </Link>
          <Link to={signedInTarget || "/register"} className="landing-primary-button">
            Bắt đầu
            <ArrowRight size={18} aria-hidden="true" />
          </Link>
        </div>
      </header>

      <main>
        <section className="landing-hero">
          <img
            src="/image/multichoice.jpg"
            alt="Phiếu trắc nghiệm đang được tô đáp án"
            className="landing-hero-image"
          />
          <div className="landing-hero-overlay" />
          <motion.div
            className="landing-hero-content"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
          >
            <div className="landing-eyebrow">
              <Sparkles size={17} aria-hidden="true" />
              <span>Chấm trắc nghiệm bằng AI</span>
            </div>
            <h1>VeritaAI</h1>
            <p>
              Quét phiếu, chấm điểm, lưu bản ghi và xuất kết quả trong một giao diện web
              gọn gàng cho giáo viên và nhà trường.
            </p>
            <div className="landing-hero-actions">
              <Link to={signedInTarget || "/register"} className="landing-primary-button landing-hero-button">
                Dùng thử hệ thống
                <ArrowRight size={18} aria-hidden="true" />
              </Link>
              <Link to={signedInTarget || "/login"} className="landing-secondary-button">
                Vào trang đăng nhập
              </Link>
            </div>
          </motion.div>
        </section>

        <section className="landing-proof" aria-label="Điểm nổi bật">
          <div>
            <strong>Camera realtime</strong>
            <span>Đọc frame qua trình duyệt</span>
          </div>
          <div>
            <strong>Nhiều mã đề</strong>
            <span>Quản lý đáp án theo bài thi</span>
          </div>
          <div>
            <strong>Xuất dữ liệu</strong>
            <span>Phục vụ tổng hợp điểm</span>
          </div>
        </section>

        <section className="landing-section" id="benefits">
          <div className="landing-section-heading">
            <span>Lợi ích</span>
            <h2>Giảm thời gian chấm bài nhưng vẫn giữ quyền kiểm soát kết quả.</h2>
          </div>
          <div className="landing-benefit-grid">
            {benefitItems.map((item) => {
              const Icon = item.icon;
              return (
                <article className="landing-card" key={item.title}>
                  <Icon className="landing-card-icon" aria-hidden="true" />
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="landing-section landing-section-alt" id="features">
          <div className="landing-section-heading">
            <span>Tính năng</span>
            <h2>Bộ công cụ tập trung cho quy trình chấm phiếu trắc nghiệm.</h2>
          </div>
          <div className="landing-feature-grid">
            {featureItems.map((item) => {
              const Icon = item.icon;
              return (
                <article className="landing-feature-card" key={item.title}>
                  <div className="landing-feature-icon">
                    <Icon size={22} aria-hidden="true" />
                  </div>
                  <div>
                    <h3>{item.title}</h3>
                    <p>{item.description}</p>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="landing-section landing-workflow-section" id="workflow">
          <div className="landing-section-heading">
            <span>Quy trình</span>
            <h2>Từ đáp án đến bảng điểm trong bốn bước rõ ràng.</h2>
          </div>
          <div className="landing-workflow">
            <ol className="landing-steps">
              {workflowItems.map((item, index) => (
                <li key={item}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <p>{item}</p>
                </li>
              ))}
            </ol>
            <div className="landing-preview" aria-label="Mô phỏng giao diện chấm bài">
              <div className="landing-preview-panel landing-preview-sheet">
                <div className="landing-preview-toolbar">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="landing-bubble-grid" aria-hidden="true">
                  {Array.from({ length: 28 }).map((_, index) => (
                    <span
                      key={index}
                      className={index === 2 || index === 7 || index === 17 || index === 24 ? "is-filled" : ""}
                    />
                  ))}
                </div>
              </div>
              <div className="landing-preview-panel landing-preview-result">
                <div>
                  <span className="landing-preview-label">Điểm</span>
                  <strong>8.75</strong>
                </div>
                <div className="landing-score-row">
                  <span>Đúng</span>
                  <b>35</b>
                </div>
                <div className="landing-score-row">
                  <span>Sai</span>
                  <b>5</b>
                </div>
                <div className="landing-score-row">
                  <span>Mã đề</span>
                  <b>102</b>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-section landing-privacy" id="privacy">
          <div className="landing-privacy-content">
            <ShieldCheck size={34} aria-hidden="true" />
            <div>
              <span>Bảo mật dữ liệu</span>
              <h2>Thiết kế cho môi trường nội bộ trường học.</h2>
              <p>
                VeritaAI dùng kết nối HTTPS khi triển khai local/LAN, tách API backend và giao diện
                frontend, đồng thời lưu dữ liệu chấm theo tài khoản người dùng.
              </p>
            </div>
          </div>
          <div className="landing-privacy-list">
            <div>
              <Lock size={19} aria-hidden="true" />
              <span>Đăng nhập trước khi quản lý bài thi</span>
            </div>
            <div>
              <Lock size={19} aria-hidden="true" />
              <span>Camera chạy trong secure context</span>
            </div>
          </div>
        </section>

        <section className="landing-final-cta">
          <h2>Sẵn sàng chấm bài với VeritaAI?</h2>
          <p>Truy cập hệ thống để tạo bài thi, quét phiếu và xuất kết quả cho lớp học.</p>
          <Link to={signedInTarget || "/register"} className="landing-primary-button">
            Tạo tài khoản
            <ArrowRight size={18} aria-hidden="true" />
          </Link>
        </section>
      </main>

      <footer className="landing-footer">
        <span>VeritaAI</span>
        <span>Hệ thống chấm trắc nghiệm bằng AI</span>
      </footer>
    </div>
  );
}
