import TopMenu from "../UI_Components/TopMenu";
import React, { useEffect, useState } from "react";
import "../UI_Components/HandwrittenQuestionPage.css";
import ImagePreviewList from "../UI_Components/ImagePreviewList";
import ImageUploader from "../UI_Components/ImageUploader.tsx";
import UploadPopup from "../UI_Components/UploadPopup.tsx";
import ViewImageModal from "../UI_Components/ViewImageModal.tsx";
import { API_CONFIG } from "../config/api";

type AnswerKeyItem = {
  ocrid: number;
  ocr_name: string;
  created_at?: string;
};

export default function HandwrittenQuestionPage() {
  const [mode, setMode] = useState<"upload" | "grade">("upload");

  const [answerKeys, setAnswerKeys] = useState<AnswerKeyItem[]>([]);
  const [selectedAnswerKeyId, setSelectedAnswerKeyId] = useState<number | null>(null);

  const [draftFileName, setDraftFileName] = useState("");
  const [draftAnswerContent, setDraftAnswerContent] = useState("");
  const [isUploadingAnswerKey, setIsUploadingAnswerKey] = useState(false);
  const [isSavingAnswerKey, setIsSavingAnswerKey] = useState(false);

  const [essayFiles, setEssayFiles] = useState<File[]>([]);
  const [essayImages, setEssayImages] = useState<string[]>([]);
  const [useAnswerCorrectness, setUseAnswerCorrectness] = useState(true);
  const [ocrModel, setOcrModel] = useState<"openai_gpt4o" | "openai_gpt4o_mini">("openai_gpt4o_mini");

  const [serverResult, setServerResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const [showPopup, setShowPopup] = useState(false);
  const [viewImage, setViewImage] = useState<string | null>(null);

  const uid = Number(localStorage.getItem("uid") || "0");

  const loadAnswerKeys = async () => {
    if (!uid) return;
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.LIST_ANSWER_KEYS(uid));
      if (!res.ok) return;
      const data = await res.json();
      const keys = data.answer_keys || [];
      setAnswerKeys(keys);
      if (!selectedAnswerKeyId && keys.length > 0) {
        setSelectedAnswerKeyId(keys[0].ocrid);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadAnswerKeys();
  }, []);

  const handleAnswerKeyFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    if (!uid) {
      setErrorMessage("Bạn chưa đăng nhập");
      return;
    }

    const file = files[0];
    const formData = new FormData();
    formData.append("uid", String(uid));
    formData.append("answer_key_file", file);

    setIsUploadingAnswerKey(true);
    setErrorMessage("");

    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.UPLOAD_ANSWER_KEY, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload đáp án thất bại");

      setDraftFileName(data.file_name || file.name);
      setDraftAnswerContent(data.ocr_answer || "");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Upload đáp án thất bại");
    } finally {
      setIsUploadingAnswerKey(false);
    }
  };

  const handleSaveAnswerKey = async () => {
    if (!uid) {
      setErrorMessage("Bạn chưa đăng nhập");
      return;
    }
    if (!draftAnswerContent.trim()) {
      setErrorMessage("Nội dung đáp án trống");
      return;
    }

    const formData = new FormData();
    formData.append("uid", String(uid));
    formData.append("ocr_name", draftFileName || "answer_key.docx");
    formData.append("ocr_answer", draftAnswerContent);

    setIsSavingAnswerKey(true);
    setErrorMessage("");
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.SAVE_ANSWER_KEY, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Lưu đáp án thất bại");

      await loadAnswerKeys();
      setSelectedAnswerKeyId(data.ocrid || null);
      alert("Đã lưu đáp án thành công");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Lưu đáp án thất bại");
    } finally {
      setIsSavingAnswerKey(false);
    }
  };

  const handleLoadOldAnswer = async (ocrid: number) => {
    if (!uid) return;
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.GET_ANSWER_KEY(ocrid, uid));
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Không tải được đáp án");

      setSelectedAnswerKeyId(data.ocrid);
      setDraftFileName(data.ocr_name);
      setDraftAnswerContent(data.ocr_answer || "");
      setMode("upload");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Không tải được đáp án");
    }
  };

  const handleDeleteAnswer = async (ocrid: number) => {
    if (!uid) return;
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.DELETE_ANSWER_KEY(ocrid, uid), {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Xóa thất bại");

      if (selectedAnswerKeyId === ocrid) setSelectedAnswerKeyId(null);
      await loadAnswerKeys();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Xóa đáp án thất bại");
    }
  };

  const handleDownloadAnswer = async (ocrid: number) => {
    if (!uid) return;
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.DOWNLOAD_ANSWER_KEY(ocrid, uid));
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Không tải được file đáp án");
      }

      const blob = await res.blob();
      const disposition = res.headers.get("content-disposition") || "";
      const fallbackName = `answer_key_${ocrid}.txt`;
      const matched = disposition.match(/filename="?([^";]+)"?/i);
      const fileName = matched?.[1] || fallbackName;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Không tải được file đáp án");
    }
  };

  const handleImages = (files: FileList | null) => {
    if (!files) return;
    const arrFiles = Array.from(files);
    const previewUrls = arrFiles.map((f) => URL.createObjectURL(f));
    setEssayFiles((prev) => [...prev, ...arrFiles]);
    setEssayImages((prev) => [...prev, ...previewUrls]);
  };

  const removeImage = (index: number) => {
    setEssayImages((prev) => prev.filter((_, i) => i !== index));
    setEssayFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    if (!uid) {
      setErrorMessage("Bạn chưa đăng nhập");
      return;
    }
    if (!selectedAnswerKeyId) {
      setErrorMessage("Vui lòng chọn đáp án đã lưu để chấm điểm");
      return;
    }
    if (essayFiles.length === 0) {
      setErrorMessage("Vui lòng tải ít nhất 1 ảnh bài làm");
      return;
    }

    const formData = new FormData();
    formData.append("uid", String(uid));
    formData.append("ocrid", String(selectedAnswerKeyId));
    formData.append("use_answer_correctness", String(useAnswerCorrectness));
    formData.append("ocr_model", ocrModel);
    essayFiles.forEach((f) => formData.append("essay_images", f));

    setIsLoading(true);
    setErrorMessage("");
    try {
      const res = await fetch(API_CONFIG.HANDWRITTEN.UPLOAD, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Chấm điểm thất bại");
      setServerResult(data);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Chấm điểm thất bại");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <TopMenu />
      <div className="handwritten-container">
        <div style={{ display: "flex", gap: "10px", marginBottom: "16px" }}>
          <button className="submit-btn" onClick={() => setMode("upload")} style={{ opacity: mode === "upload" ? 1 : 0.7 }}>
            Tải đáp án
          </button>
          <button className="submit-btn" onClick={() => setMode("grade")} style={{ opacity: mode === "grade" ? 1 : 0.7 }}>
            Chấm điểm
          </button>
        </div>

        {mode === "upload" && (
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "16px" }}>
            <div style={{ padding: "16px", border: "2px solid #1976d2", borderRadius: "10px", backgroundColor: "#e3f2fd" }}>
              <h2 style={{ marginTop: 0 }}>Tải đáp án</h2>
              <input type="file" accept=".doc,.docx,.pdf,.txt" onChange={handleAnswerKeyFileChange} disabled={isUploadingAnswerKey} />
              {isUploadingAnswerKey && <p>Đang bóc tách nội dung đáp án...</p>}

              {draftAnswerContent && (
                <div style={{ marginTop: "12px" }}>
                  <p><strong>File:</strong> {draftFileName}</p>
                  <p><strong>Nội dung dùng làm đáp án:</strong></p>
                  <textarea
                    value={draftAnswerContent}
                    onChange={(e) => setDraftAnswerContent(e.target.value)}
                    rows={18}
                    style={{ width: "100%", resize: "vertical", padding: "10px", borderRadius: "8px", border: "1px solid #90caf9" }}
                  />
                  <button className="submit-btn" onClick={handleSaveAnswerKey} disabled={isSavingAnswerKey} style={{ marginTop: "10px" }}>
                    {isSavingAnswerKey ? "Đang lưu..." : "Lưu đáp án"}
                  </button>
                </div>
              )}
            </div>

            <div style={{ padding: "16px", border: "1px solid #ddd", borderRadius: "10px", backgroundColor: "#fafafa" }}>
              <h3 style={{ marginTop: 0 }}>Danh sách đáp án đã tải</h3>
              {answerKeys.length === 0 && <p>Chưa có đáp án nào</p>}
              {answerKeys.map((item) => (
                <div key={item.ocrid} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", marginBottom: "10px", backgroundColor: "#fff" }}>
                  <div style={{ fontWeight: "bold" }}>{item.ocr_name}</div>
                  <div style={{ fontSize: "12px", color: "#666", marginBottom: "8px" }}>
                    {item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : ""}
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button onClick={() => handleLoadOldAnswer(item.ocrid)}>Tải đáp án cũ</button>
                    <button onClick={() => handleDownloadAnswer(item.ocrid)}>Tải file</button>
                    <button onClick={() => handleDeleteAnswer(item.ocrid)}>Xóa</button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {mode === "grade" && (
          <div>
            <div style={{ padding: "16px", border: "2px solid #4caf50", borderRadius: "10px", backgroundColor: "#e8f5e9", marginBottom: "12px" }}>
              <h2 style={{ marginTop: 0 }}>Chấm điểm</h2>
              <label style={{ display: "block", marginBottom: "8px" }}>Chọn đáp án đã lưu:</label>
              <select
                value={selectedAnswerKeyId || ""}
                onChange={(e) => setSelectedAnswerKeyId(e.target.value ? Number(e.target.value) : null)}
                style={{ width: "100%", padding: "10px", borderRadius: "6px", border: "1px solid #bbb", marginBottom: "10px" }}
              >
                <option value="">-- Chọn đáp án --</option>
                {answerKeys.map((item) => (
                  <option key={item.ocrid} value={item.ocrid}>
                    #{item.ocrid} - {item.ocr_name}
                  </option>
                ))}
              </select>

              <label style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <input
                  type="checkbox"
                  checked={useAnswerCorrectness}
                  onChange={(e) => setUseAnswerCorrectness(e.target.checked)}
                />
                Dùng Answer Correctness cho từng câu
              </label>

              <label style={{ display: "block", marginTop: "10px", marginBottom: "6px" }}>Model OCR LLM:</label>
              <select
                value={ocrModel}
                onChange={(e) => setOcrModel(e.target.value as "openai_gpt4o" | "openai_gpt4o_mini")}
                style={{ width: "100%", padding: "10px", borderRadius: "6px", border: "1px solid #bbb" }}
              >
                <option value="openai_gpt4o_mini">GPT-4o-mini (nhanh, tiết kiệm)</option>
                <option value="openai_gpt4o">GPT-4o (độ chính xác cao hơn)</option>
              </select>
            </div>

            <div className="upload-area">
              <ImageUploader onClick={() => selectedAnswerKeyId && setShowPopup(true)} />
              <ImagePreviewList images={essayImages} onRemove={(idx: number) => removeImage(idx)} onView={(img: string) => setViewImage(img)} />
            </div>

            <div className="submit-area">
              <button className="submit-btn" onClick={handleSubmit} disabled={isLoading}>
                {isLoading ? "Đang xử lý..." : "Chấm điểm"}
              </button>
            </div>
          </div>
        )}
      </div>

      {showPopup && (
        <UploadPopup
          onSelect={(files: FileList) => {
            handleImages(files);
            setShowPopup(false);
          }}
          onClose={() => setShowPopup(false)}
        />
      )}

      {viewImage && <ViewImageModal img={viewImage} onClose={() => setViewImage(null)} />}

      {errorMessage && (
        <div style={{ color: "#f44336", fontSize: "14px", padding: "10px", backgroundColor: "#ffebee", borderRadius: "4px", margin: "10px auto", maxWidth: "1100px", textAlign: "center" }}>
          {errorMessage}
        </div>
      )}

      {serverResult && serverResult.results && (
        <div className="result-box">
          <h3>Kết quả chấm điểm</h3>
          {serverResult.results.map((essayResult: any, essayIndex: number) => {
            const totalMax = essayResult.total_max_score || (essayResult.questions?.length || 0) * 10;
            return (
              <div key={essayIndex} style={{ border: "2px solid #bcccdc", padding: "20px", marginBottom: "20px", borderRadius: "10px", backgroundColor: "#f8fbff", color: "#102a43" }}>
                <h4 style={{ marginTop: 0 }}>Bài {essayIndex + 1}: {essayResult.file}</h4>
                <div style={{ fontSize: "24px", fontWeight: "bold", textAlign: "center", marginBottom: "12px", color: "#0b3d91" }}>
                  Tổng điểm: {essayResult.total_score?.toFixed?.(1) ?? essayResult.total_score} / {totalMax}
                </div>

                <div style={{ border: "1px solid #d9e2ec", padding: "12px", marginBottom: "12px", borderRadius: "8px", backgroundColor: "#ffffff" }}>
                  <p style={{ margin: "0 0 8px 0" }}><strong>1. Nội dung bài làm đã trích xuất:</strong></p>
                  <pre style={{ backgroundColor: "#f8fafc", color: "#102a43", border: "1px solid #d9e2ec", padding: "10px", borderRadius: "6px", whiteSpace: "pre-wrap" }}>
                    {essayResult.recognized_text || "(không có nội dung OCR)"}
                  </pre>
                </div>

                {essayResult.questions?.map((q: any, idx: number) => (
                  <div key={idx} style={{ border: "1px solid #d9e2ec", padding: "12px", marginBottom: "10px", borderRadius: "8px", backgroundColor: "#ffffff", color: "#243b53" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                      <strong>{q.id}</strong>
                      <strong>{q.score} / {q.max_score}</strong>
                    </div>
                    <p style={{ margin: "6px 0" }}><strong>Bài làm đọc từ ảnh:</strong></p>
                    <pre style={{ backgroundColor: "#f5f7fa", color: "#102a43", border: "1px solid #d9e2ec", padding: "10px", borderRadius: "6px", whiteSpace: "pre-wrap" }}>{q.content || "(trống)"}</pre>
                    <p style={{ margin: "8px 0 4px 0" }}><strong>Đáp án:</strong></p>
                    <pre style={{ backgroundColor: "#f8fafc", color: "#102a43", border: "1px solid #d9e2ec", padding: "10px", borderRadius: "6px", whiteSpace: "pre-wrap" }}>{q.answer_key_used || ""}</pre>
                    <p style={{ marginTop: "8px", color: "#1f3a8a", fontWeight: 600 }}>
                      3. Điểm số (Answer Correctness Score): {q.score ?? 0} / {q.max_score ?? 10}
                    </p>
                    <p style={{ marginTop: "4px", color: "#1f3a8a", fontWeight: 600 }}>
                      Answer Correctness: {typeof q.answer_correctness === "number" ? `${(q.answer_correctness * 100).toFixed(1)}%` : "N/A"}
                    </p>
                    <div style={{ marginTop: "8px", border: "1px solid #d9e2ec", borderRadius: "8px", padding: "10px", backgroundColor: "#f8fbff" }}>
                      <p style={{ margin: "0 0 6px 0", color: "#102a43", fontWeight: 700 }}>
                        2. Phân tích chi tiết (RAGAS Analysis):
                      </p>
                      <p style={{ margin: "6px 0", color: "#102a43" }}>
                        <strong>Ý đúng (TP):</strong> {(q.ragas_analysis?.true_positives || []).length > 0 ? (q.ragas_analysis?.true_positives || []).join("; ") : "Không có"}
                      </p>
                      <p style={{ margin: "6px 0", color: "#102a43" }}>
                        <strong>Ý sai/thừa (FP):</strong> {(q.ragas_analysis?.false_positives || []).length > 0 ? (q.ragas_analysis?.false_positives || []).join("; ") : "Không có"}
                      </p>
                      <p style={{ margin: "6px 0", color: "#102a43" }}>
                        <strong>Ý thiếu (FN):</strong> {(q.ragas_analysis?.false_negatives || []).length > 0 ? (q.ragas_analysis?.false_negatives || []).join("; ") : "Không có"}
                      </p>
                    </div>
                    <p style={{ marginTop: "4px", color: "#1e3a5f", lineHeight: 1.5 }}>
                      <strong>4. Lý do chấm điểm (Answer Correctness Reason):</strong> {q.answer_correctness_reason || q.feedback || "Không có"}
                    </p>
                    <div style={{ marginTop: "8px" }}>
                      <p style={{ margin: "0 0 6px 0", color: "#102a43", fontWeight: 700 }}>
                        JSON Output:
                      </p>
                      <pre style={{ backgroundColor: "#f8fafc", color: "#102a43", border: "1px solid #d9e2ec", padding: "10px", borderRadius: "6px", whiteSpace: "pre-wrap" }}>
                        {JSON.stringify(q.strict_json_output || {
                          extracted_text: q.content || "",
                          ragas_analysis: {
                            true_positives: q.ragas_analysis?.true_positives || [],
                            false_positives: q.ragas_analysis?.false_positives || [],
                            false_negatives: q.ragas_analysis?.false_negatives || [],
                          },
                          answer_correctness_score: q.score || 0,
                          answer_correctness_reason: q.answer_correctness_reason || q.feedback || "",
                        }, null, 2)}
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
