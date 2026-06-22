import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_CONFIG } from "../config/api";
import OmrProfileRoiEditor from "../features/omr/components/OmrProfileRoiEditor";
import "../features/omr/styles/OmrMobileApp.css";
import "./AdminPage.css";
import type { FormProfile } from "../features/omr/types";

const DEFAULT_CREATE_FIELDS = {
  title: "",
  default_questions: 40,
  num_choices: 4,
  rows_per_block: 20,
  student_id_digits: 6,
  sid_has_write_row: true,
};

export default function AdminPage() {
  const navigate = useNavigate();


  const [formProfiles, setFormProfiles] = useState<FormProfile[]>([]);
  const [formProfilesLoading, setFormProfilesLoading] = useState(false);
  const [roiEditorProfile, setRoiEditorProfile] = useState<FormProfile | null>(null);
  const [samplePreview, setSamplePreview] = useState<{ title: string; fileName: string; url: string } | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // Create form state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createFile, setCreateFile] = useState<File | null>(null);
  const [createFields, setCreateFields] = useState(DEFAULT_CREATE_FIELDS);
  const [creating, setCreating] = useState(false);

  const buildSampleUrl = (sampleFile: string) =>
    `${API_CONFIG.BASE_URL}/static/omr_data/${encodeURIComponent(sampleFile)}`;

  const loadFormProfiles = useCallback(async () => {
    setFormProfilesLoading(true);
    try {
      const res = await fetch(API_CONFIG.OMR.LIST_FORM_PROFILES);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Không tải được profile phiếu mẫu");
      const rows: FormProfile[] = Array.isArray(data.profiles) ? data.profiles : [];
      setFormProfiles(rows);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Không tải được profile phiếu mẫu.");
    } finally {
      setFormProfilesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadFormProfiles();
  }, [loadFormProfiles]);

  function handleProfileSaved(nextProfile: FormProfile) {
    setFormProfiles((prev) => prev.map((item) => (item.code === nextProfile.code ? nextProfile : item)));
  }

  async function handleCreateProfile() {
    if (!createFile) return;
    setCreating(true);
    try {
      // Step 1: upload sample file
      const fd = new FormData();
      fd.append("file", createFile);
      const upRes = await fetch(API_CONFIG.OMR.UPLOAD_FORM_SAMPLE, { method: "POST", body: fd });
      const upData = await upRes.json();
      if (!upRes.ok) throw new Error(upData.detail || "Upload file thất bại");
      const sampleFile: string = upData.filename;

      // Step 2: create profile
      const profileRes = await fetch(API_CONFIG.OMR.SAVE_FORM_PROFILE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_file: sampleFile,
          title: createFields.title || sampleFile.replace(/\.[^.]+$/, ""),
          default_questions: createFields.default_questions,
          num_choices: createFields.num_choices,
          rows_per_block: createFields.rows_per_block,
          student_id_digits: createFields.student_id_digits,
          sid_has_write_row: createFields.sid_has_write_row,
        }),
      });
      const profileData = await profileRes.json();
      if (!profileRes.ok) throw new Error(profileData.detail || "Tạo profile thất bại");

      setSuccessMessage(`Đã tạo mẫu phiếu "${profileData.profile?.title || sampleFile}"`);
      setShowCreateForm(false);
      setCreateFile(null);
      setCreateFields(DEFAULT_CREATE_FIELDS);
      await loadFormProfiles();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Lỗi khi tạo profile");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="admin-page-shell">
      <header className="admin-header">
        <button className="admin-back-btn tap-feedback" onClick={() => navigate(-1)}>
          ← Quay lại
        </button>
        <h1 className="admin-title">Quản lý mẫu phiếu OMR</h1>
      </header>

      <main className="admin-main">
        {/* Create new profile button & form */}
        <div className="admin-create-bar">
          <button
            className="primary-btn tap-feedback"
            onClick={() => setShowCreateForm((v) => !v)}
          >
            {showCreateForm ? "✕ Đóng" : "+ Thêm mẫu phiếu mới"}
          </button>
        </div>

        {showCreateForm && (
          <section className="omr-section admin-create-form">
            <h4 className="home-list-title">Thêm mẫu phiếu mới</h4>

            <div className="admin-field">
              <label>File mẫu (PDF / PNG / JPG / WebP)</label>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.webp"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null;
                  setCreateFile(f);
                  if (f) {
                    setCreateFields((p) => ({
                      ...p,
                      title: f.name.replace(/\.[^.]+$/, ""),
                    }));
                  }
                }}
              />
              {createFile && <div className="admin-file-name">📄 {createFile.name}</div>}
            </div>

            <div className="admin-field">
              <label>Tên hiển thị</label>
              <input
                type="text"
                value={createFields.title}
                onChange={(e) => setCreateFields((p) => ({ ...p, title: e.target.value }))}
                placeholder="VD: A4-40 câu"
              />
            </div>

            <div className="admin-field admin-field-row">
              <div>
                <label>Số câu hỏi</label>
                <input
                  type="number"
                  min={1}
                  max={300}
                  value={createFields.default_questions}
                  onChange={(e) =>
                    setCreateFields((p) => ({ ...p, default_questions: Math.max(1, Number(e.target.value)) }))
                  }
                />
              </div>
              <div>
                <label>Số lựa chọn / câu</label>
                <select
                  value={createFields.num_choices}
                  onChange={(e) => setCreateFields((p) => ({ ...p, num_choices: Number(e.target.value) }))}
                >
                  <option value={2}>2 (A–B)</option>
                  <option value={4}>4 (A–D)</option>
                  <option value={5}>5 (A–E)</option>
                </select>
              </div>
            </div>

            <div className="admin-field admin-field-row">
              <div>
                <label>Số câu mỗi block</label>
                <input
                  type="number"
                  min={1}
                  max={60}
                  value={createFields.rows_per_block}
                  onChange={(e) =>
                    setCreateFields((p) => ({ ...p, rows_per_block: Math.max(1, Number(e.target.value)) }))
                  }
                />
              </div>
              <div>
                <label>Số chữ số MSSV</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={createFields.student_id_digits}
                  onChange={(e) =>
                    setCreateFields((p) => ({ ...p, student_id_digits: Math.max(1, Number(e.target.value)) }))
                  }
                />
              </div>
            </div>

            <div className="admin-field admin-field-check">
              <label>
                <input
                  type="checkbox"
                  checked={createFields.sid_has_write_row}
                  onChange={(e) => setCreateFields((p) => ({ ...p, sid_has_write_row: e.target.checked }))}
                />
                Có hàng ghi tay MSSV ở đầu
              </label>
            </div>

            <button
              className="primary-btn tap-feedback"
              disabled={!createFile || creating}
              onClick={handleCreateProfile}
            >
              {creating ? "Đang tạo…" : "Tạo mẫu phiếu"}
            </button>
          </section>
        )}

        {/* Profile list */}
        <section className="omr-section">
          <h4 className="home-list-title">Danh sách Form Profile</h4>

          {formProfilesLoading && (
            <div className="empty-box">Đang tải profile phiếu mẫu...</div>
          )}
          {!formProfilesLoading && formProfiles.length === 0 && (
            <div className="empty-box">Chưa có profile phiếu mẫu trong omr_data.</div>
          )}
          {!formProfilesLoading &&
            formProfiles.map((p) => (
              <div
                key={`profile_${p.code}`}
                className="template-card tap-feedback"
                role="button"
                tabIndex={0}
                onClick={() =>
                  setSamplePreview({
                    title: p.title,
                    fileName: p.sample_file,
                    url: buildSampleUrl(p.sample_file),
                  })
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSamplePreview({
                      title: p.title,
                      fileName: p.sample_file,
                      url: buildSampleUrl(p.sample_file),
                    });
                  }
                }}
              >
                <div>
                  <div className="test-title">{p.title}</div>
                  <div className="test-sub">Mã đề: 3 số</div>
                  <div className="test-sub">
                    MSSV: {Math.max(1, Number(p.student_id_digits || 6))} số
                  </div>
                  <div className="test-sub">
                    Câu mặc định: {p.default_questions} | Điểm mặc định: 10
                  </div>
                </div>
                <div className="template-actions">
                  <button
                    type="button"
                    className="mini-btn tap-feedback"
                    onClick={(e) => {
                      e.stopPropagation();
                      setRoiEditorProfile(p);
                    }}
                  >
                    Cấu hình ROI
                  </button>
                  <button
                    type="button"
                    className="mini-btn tap-feedback"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSamplePreview({
                        title: p.title,
                        fileName: p.sample_file,
                        url: buildSampleUrl(p.sample_file),
                      });
                    }}
                  >
                    Xem mẫu
                  </button>
                  <a
                    className="mini-btn tap-feedback template-download-btn"
                    href={buildSampleUrl(p.sample_file)}
                    download={p.sample_file}
                    onClick={(e) => e.stopPropagation()}
                  >
                    Tải mẫu
                  </a>
                </div>
              </div>
            ))}
        </section>
      </main>

      {/* Sample preview overlay */}
      {samplePreview && (
        <div className="sample-preview-overlay" onClick={() => setSamplePreview(null)}>
          <div className="sample-preview-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="sample-preview-head">
              <span className="sample-preview-title">{samplePreview.title}</span>
              <button
                className="header-back-btn tap-feedback"
                onClick={() => setSamplePreview(null)}
              >
                ✕
              </button>
            </div>
            <div className="sample-preview-body">
              {samplePreview.fileName.toLowerCase().endsWith(".pdf") ? (
                <iframe src={samplePreview.url} title={`sample_${samplePreview.fileName}`} />
              ) : (
                <img src={samplePreview.url} alt={`Mẫu ${samplePreview.title}`} />
              )}
            </div>
          </div>
        </div>
      )}

      {roiEditorProfile && (
        <OmrProfileRoiEditor
          profile={roiEditorProfile}
          sampleUrl={buildSampleUrl(roiEditorProfile.sample_file)}
          onClose={() => setRoiEditorProfile(null)}
          onSaved={handleProfileSaved}
          onError={setErrorMessage}
          onSuccess={setSuccessMessage}
        />
      )}

      {!!errorMessage && (
        <div className="error-toast" onClick={() => setErrorMessage("")}>
          {errorMessage}
        </div>
      )}
      {!!successMessage && (
        <div className="success-toast" onClick={() => setSuccessMessage("")}>
          {successMessage}
        </div>
      )}
    </div>
  );
}
