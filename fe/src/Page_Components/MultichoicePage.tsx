import TopMenu from "../UI_Components/TopMenu";
import { useEffect, useState } from "react";
import "../UI_Components/HandwrittenQuestionPage.css";
import ImageUploader from "../UI_Components/ImageUploader.tsx";
import UploadPopup from "../UI_Components/UploadPopup.tsx";
import ViewImageModal from "../UI_Components/ViewImageModal.tsx";
import OmrQuadCropModal from "../UI_Components/OmrQuadCropModal.tsx";
import { API_CONFIG } from "../config/api";

type CropPoint = { x: number; y: number };

const buildDefaultCropQuad = (): CropPoint[] => [
    { x: 0.08, y: 0.08 },
    { x: 0.92, y: 0.08 },
    { x: 0.92, y: 0.92 },
    { x: 0.08, y: 0.92 },
];

export default function MultichoicePage() {
    const inputStyle = { width: "100%", padding: "8px", marginTop: "5px" } as const;

    const [activeMode, setActiveMode] = useState<"template" | "grading" | "vault">("template");
    // --- STATE QUẢN LÝ ẢNH ---
    const [omrFiles, setOmrFiles] = useState<File[]>([]);
    const [omrImagePreviews, setOmrImagePreviews] = useState<string[]>([]);

    // --- STATE CẤU HÌNH CHẤM ---
    const [numQuestions, setNumQuestions] = useState<number>(80);
    const [numChoices, setNumChoices] = useState<number>(5);
    const [rowsPerBlock, setRowsPerBlock] = useState<number>(20);
    const [examTitle, setExamTitle] = useState<string>("OMR Practice Exam");
    const [studentIdDigits, setStudentIdDigits] = useState<number>(6);
    const [omrCode, setOmrCode] = useState<string>("001");
    const [templateInfoFields, setTemplateInfoFields] = useState<string[]>(["Tên", "Lớp"]);
    const [templateAnswerKey, setTemplateAnswerKey] = useState<string>("");
    const [templateAnswerKeyFile, setTemplateAnswerKeyFile] = useState<File | null>(null);

    // --- STATE TẠO PHIẾU OMR ---
    const [templateImageUrl, setTemplateImageUrl] = useState<string | null>(null);
    const [isGeneratingTemplate, setIsGeneratingTemplate] = useState<boolean>(false);
    const [templateDraft, setTemplateDraft] = useState<any>(null);
    const [isSavingTemplate, setIsSavingTemplate] = useState<boolean>(false);
    const [vaultItems, setVaultItems] = useState<any[]>([]);
    const [isLoadingVault, setIsLoadingVault] = useState<boolean>(false);

    // --- STATE KẾT QUẢ ---
    const [serverResult, setServerResult] = useState<any>(null);
    const [batchResults, setBatchResults] = useState<any[]>([]);
    const [batchZipUrl, setBatchZipUrl] = useState<string | null>(null);
    const [batchFilter, setBatchFilter] = useState<"all" | "failed">("all");
    const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [errorMessage, setErrorMessage] = useState<string>("");

    // State Popup
    const [showPopup, setShowPopup] = useState<boolean>(false);
    const [viewImage, setViewImage] = useState<string | null>(null);
    const [manualCropEnabled, setManualCropEnabled] = useState<boolean>(true);
    const [cropQuad, setCropQuad] = useState<CropPoint[]>(buildDefaultCropQuad());
    const [showCropModal, setShowCropModal] = useState<boolean>(false);
    const [isSuggestingCrop, setIsSuggestingCrop] = useState<boolean>(false);

    const isBatchMode = omrFiles.length > 1;
    const templateInfoOptions = ["Tên", "Lớp", "Môn thi", "Lớp thi", "Năm học"];

    const toggleTemplateInfoField = (field: string) => {
        setTemplateInfoFields((prev) => {
            if (prev.includes(field)) {
                return prev.filter((x) => x !== field);
            }
            return [...prev, field];
        });
    };

    const handleDownloadFile = async (url: string, fallbackName: string) => {
        try {
            const res = await fetch(url);
            if (!res.ok) {
                throw new Error("Không tải được file");
            }
            const blob = await res.blob();
            const objectUrl = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = objectUrl;
            a.download = fallbackName;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(objectUrl);
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Tải file thất bại");
        }
    };

    const formatTemplateApiError = (rawMessage: string) => {
        const message = String(rawMessage || "").toLowerCase();
        if (
            message.includes("đề đã tồn tại") ||
            message.includes("trùng") ||
            message.includes("same")
        ) {
            return "Đề bị trùng. Vui lòng đổi Tên bài thi, Mã đề hoặc Số chữ số SID.";
        }
        return rawMessage || "Thao tác thất bại";
    };

    const loadVault = async () => {
        const uidStr = localStorage.getItem("uid");
        const uid = uidStr ? Number(uidStr) : 0;
        if (!uid) {
            setVaultItems([]);
            return;
        }
        setIsLoadingVault(true);
        try {
            const res = await fetch(API_CONFIG.OMR.LIST_TESTS(uid));
            const data = await res.json();
            if (res.ok) {
                setVaultItems(Array.isArray(data.omr_tests) ? data.omr_tests : []);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsLoadingVault(false);
        }
    };

    useEffect(() => {
        loadVault();
    }, []);

    const handleSaveTemplate = async () => {
        if (!templateDraft) {
            setErrorMessage("Chưa có phiếu nháp để lưu.");
            return;
        }
        setIsSavingTemplate(true);
        setErrorMessage("");
        try {
            const form = new FormData();
            form.append("uid", String(templateDraft.uid));
            form.append("omr_name", String(templateDraft.omr_name));
            form.append("omr_code", String(templateDraft.omr_code));
            form.append("omr_quest", String(templateDraft.omr_quest));
            form.append("omr_answer", JSON.stringify(templateDraft.omr_answer || []));
            form.append("template_image", String(templateDraft.template_image || ""));
            form.append("info_fields", JSON.stringify(templateDraft.info_fields || []));
            form.append("options", String(templateDraft.options || numChoices));
            form.append("rows_per_block", String(templateDraft.rows_per_block || rowsPerBlock));
            form.append("student_id_digits", String(templateDraft.student_id_digits || studentIdDigits));

            const res = await fetch(API_CONFIG.OMR.SAVE_TEMPLATE, {
                method: "POST",
                body: form,
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(formatTemplateApiError(data.detail || data.message || "Lưu phiếu thất bại"));
            }
            setTemplateDraft(null);
            await loadVault();
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Lưu phiếu thất bại");
        } finally {
            setIsSavingTemplate(false);
        }
    };

    const handleDeleteVaultItem = async (omrid: number) => {
        const uidStr = localStorage.getItem("uid");
        const uid = uidStr ? Number(uidStr) : 0;
        if (!uid) return;
        if (!confirm("Bạn chắc chắn muốn xóa phiếu này?")) return;
        try {
            const res = await fetch(API_CONFIG.OMR.DELETE_TEST(uid, omrid), { method: "DELETE" });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || data.message || "Xóa thất bại");
            await loadVault();
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Xóa thất bại");
        }
    };

    // Xử lý khi chọn ảnh
    const handleImages = async (files: FileList | null) => {
        if (!files || files.length === 0) return;

        const selectedFilesRaw = Array.from(files);
        const selectedFiles = selectedFilesRaw.slice(0, 50);
        const selectedPreviews = selectedFiles.map((f) => URL.createObjectURL(f));
        setOmrFiles(selectedFiles);
        setOmrImagePreviews(selectedPreviews);

        if (selectedFilesRaw.length > 50) {
            setErrorMessage("Mỗi lần gửi tối đa 50 ảnh. Hệ thống đã giữ 50 ảnh đầu tiên.");
        } else {
            setErrorMessage("");
        }
        
        // Reset kết quả cũ
        setServerResult(null);
        setBatchResults([]);
        setBatchZipUrl(null);
        setBatchFilter("all");
        setResultImageUrl(null);
        setCropQuad(buildDefaultCropQuad());

        if (manualCropEnabled && selectedFiles.length === 1) {
            setIsSuggestingCrop(true);
            try {
                const suggestForm = new FormData();
                suggestForm.append("file", selectedFiles[0]);

                const suggestRes = await fetch(API_CONFIG.OMR.SUGGEST_CROP, {
                    method: "POST",
                    body: suggestForm,
                });
                const suggestData = await suggestRes.json();

                if (suggestRes.ok && suggestData?.quad) {
                    const q = suggestData.quad;
                    const nextQuad: CropPoint[] = [
                        { x: Number(q.tl?.x ?? 0.08), y: Number(q.tl?.y ?? 0.08) },
                        { x: Number(q.tr?.x ?? 0.92), y: Number(q.tr?.y ?? 0.08) },
                        { x: Number(q.br?.x ?? 0.92), y: Number(q.br?.y ?? 0.92) },
                        { x: Number(q.bl?.x ?? 0.08), y: Number(q.bl?.y ?? 0.92) },
                    ];
                    setCropQuad(nextQuad);
                }
            } catch (err) {
                console.warn("Suggest crop failed, fallback to default quad", err);
            } finally {
                setIsSuggestingCrop(false);
                setShowCropModal(true);
            }
        } else {
            setShowCropModal(false);
        }
    };

    const removeImage = (idx: number) => {
        const nextFiles = omrFiles.filter((_, i) => i !== idx);
        const nextPreviews = omrImagePreviews.filter((_, i) => i !== idx);
        setOmrFiles(nextFiles);
        setOmrImagePreviews(nextPreviews);
        setServerResult(null);
        setBatchResults([]);
        setBatchZipUrl(null);
        setBatchFilter("all");
        setResultImageUrl(null);
        setCropQuad(buildDefaultCropQuad());
        setShowCropModal(false);
    };

    const handleSubmit = async () => {
        const uidStr = localStorage.getItem("uid");
        const uid = uidStr ? Number(uidStr) : 0;
        if (!uid) {
            setErrorMessage("Bạn chưa đăng nhập!");
            return;
        }

        if (!omrFiles.length) {
            setErrorMessage("Vui lòng tải lên ít nhất 1 ảnh phiếu trả lời!");
            return;
        }
        if (omrFiles.length > 50) {
            setErrorMessage("Mỗi lần gửi tối đa 50 ảnh.");
            return;
        }
        setIsLoading(true);
        setErrorMessage("");
        const formData = new FormData();

        if (isBatchMode) {
            omrFiles.forEach((f) => formData.append("files", f));
        } else {
            formData.append("file", omrFiles[0]);
        }
        formData.append("uid", String(uid));
        formData.append("num_questions", numQuestions.toString());
        formData.append("num_choices", numChoices.toString());
        formData.append("rows_per_block", rowsPerBlock.toString());
        formData.append("student_id_digits", studentIdDigits.toString());
        formData.append("sid_has_write_row", "true");
        if (!isBatchMode && manualCropEnabled && cropQuad.length === 4) {
            formData.append("crop_tl_x", cropQuad[0].x.toString());
            formData.append("crop_tl_y", cropQuad[0].y.toString());
            formData.append("crop_tr_x", cropQuad[1].x.toString());
            formData.append("crop_tr_y", cropQuad[1].y.toString());
            formData.append("crop_br_x", cropQuad[2].x.toString());
            formData.append("crop_br_y", cropQuad[2].y.toString());
            formData.append("crop_bl_x", cropQuad[3].x.toString());
            formData.append("crop_bl_y", cropQuad[3].y.toString());
        }
        try {
            const res = await fetch(isBatchMode ? API_CONFIG.OMR.GRADE_BATCH : API_CONFIG.OMR.GRADE, {
                method: "POST",
                body: formData,
            });

            const data = await res.json();

            if (res.ok) {
                if (isBatchMode) {
                    setServerResult(null);
                    setResultImageUrl(null);
                    setBatchResults(Array.isArray(data.results) ? data.results : []);
                    setBatchZipUrl(data.zip_url ? `${API_CONFIG.BASE_URL}${data.zip_url}` : null);
                } else {
                    setBatchResults([]);
                    setBatchZipUrl(null);
                    setServerResult(data.data);
                    setResultImageUrl(data.image_url ? `${API_CONFIG.BASE_URL}${data.image_url}` : null);
                }
            } else {
                throw new Error(data.detail || data.message || "Grading failed");
            }
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Grading failed!");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleCreateTemplate = async () => {
        const uidStr = localStorage.getItem("uid");
        const uid = uidStr ? Number(uidStr) : 0;
        if (!uid) {
            setErrorMessage("Bạn chưa đăng nhập!");
            return;
        }

        if (!examTitle.trim()) {
            setErrorMessage("Vui lòng nhập Exam Title!");
            return;
        }
        if (!/^\d{3}$/.test(omrCode.trim())) {
            setErrorMessage("Mã đề phải gồm đúng 3 chữ số (0-9)");
            return;
        }
        if (!templateAnswerKeyFile && !templateAnswerKey.trim()) {
            setErrorMessage("Vui lòng nhập đáp án hoặc tải file đáp án để tạo phiếu!");
            return;
        }

        setIsGeneratingTemplate(true);
        setErrorMessage("");

        const formData = new FormData();
        formData.append("uid", String(uid));
        formData.append("exam_title", examTitle.trim());
        formData.append("omr_code", omrCode.trim());
        formData.append("info_fields", JSON.stringify(templateInfoFields));
        formData.append("answers", templateAnswerKey);
        if (templateAnswerKeyFile) {
            formData.append("answer_key_file", templateAnswerKeyFile);
        }
        formData.append("total_questions", numQuestions.toString());
        formData.append("options", numChoices.toString());
        formData.append("student_id_digits", studentIdDigits.toString());
        formData.append("rows_per_block", rowsPerBlock.toString());

        try {
            const res = await fetch(API_CONFIG.OMR.CREATE_TEMPLATE, {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            if (!res.ok) {
                throw new Error(formatTemplateApiError(data.detail || data.message || "Template generation failed"));
            }

            if (data.template_url) {
                setTemplateImageUrl(`${API_CONFIG.BASE_URL}${data.template_url}`);
            } else {
                setTemplateImageUrl(null);
            }
            setTemplateDraft(data.draft || null);

        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Template generation failed!");
            console.error(err);
        } finally {
            setIsGeneratingTemplate(false);
        }
    };

    return (
        <div>
            <TopMenu />
            <div className="handwritten-container">
                <h1 style={{ textAlign: "center", marginBottom: "20px" }}>Thi Trắc Nghiệm (OMR)</h1>

                <div style={{ display: "flex", justifyContent: "center", marginBottom: "16px", gap: "8px" }}>
                    <button
                        className="submit-btn"
                        onClick={() => setActiveMode("template")}
                        style={{
                            width: "220px",
                            opacity: activeMode === "template" ? 1 : 0.75,
                            background: activeMode === "template" ? "#1976d2" : "#607d8b",
                        }}
                    >
                        Tạo phiếu
                    </button>
                    <button
                        className="submit-btn"
                        onClick={() => setActiveMode("grading")}
                        style={{
                            width: "220px",
                            opacity: activeMode === "grading" ? 1 : 0.75,
                            background: activeMode === "grading" ? "#2e7d32" : "#607d8b",
                        }}
                    >
                        Chấm điểm
                    </button>
                    <button
                        className="submit-btn"
                        onClick={() => {
                            setActiveMode("vault");
                            loadVault();
                        }}
                        style={{
                            width: "220px",
                            opacity: activeMode === "vault" ? 1 : 0.75,
                            background: activeMode === "vault" ? "#6d4c41" : "#607d8b",
                        }}
                    >
                        Kho phiếu
                    </button>
                </div>

                <div style={{ display: "flex", gap: "30px", justifyContent: "center", flexWrap: "wrap" }}>
                    
                    {/* CỘT TRÁI: CẤU HÌNH & UPLOAD */}
                    <div style={{ flex: 1, minWidth: "300px", maxWidth: "500px" }}>
                        
                        {activeMode === "template" && (
                        <div className="config-section" style={{ marginBottom: "20px", padding: "15px", border: "1px solid #ddd", borderRadius: "8px" }}>
                            <h3>1. Tạo phiếu khoanh đáp án OMR</h3>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Tên bài thi:</label>
                                <input
                                    type="text"
                                    value={examTitle}
                                    onChange={(e) => setExamTitle(e.target.value)}
                                    placeholder="Ví dụ: Kiểm tra giữa kỳ Toán"
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Mã đề (3 chữ số):</label>
                                <input
                                    type="text"
                                    value={omrCode}
                                    maxLength={3}
                                    onChange={(e) => setOmrCode(e.target.value.replace(/\D/g, "").slice(0, 3))}
                                    placeholder="Ví dụ: 001"
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Bảng thông tin cạnh mã đề:</label>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginTop: "6px" }}>
                                    {templateInfoOptions.map((field) => (
                                        <label key={field} style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "14px" }}>
                                            <input
                                                type="checkbox"
                                                checked={templateInfoFields.includes(field)}
                                                onChange={() => toggleTemplateInfoField(field)}
                                            />
                                            {field}
                                        </label>
                                    ))}
                                </div>
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Tổng số câu hỏi:</label>
                                <input
                                    type="number"
                                    min={1}
                                    value={numQuestions}
                                    onChange={(e) => setNumQuestions(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số lựa chọn mỗi câu:</label>
                                <input
                                    type="number"
                                    min={2}
                                    value={numChoices}
                                    onChange={(e) => setNumChoices(Math.max(2, Number(e.target.value) || 2))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số chữ số SID:</label>
                                <input
                                    type="number"
                                    min={1}
                                    value={studentIdDigits}
                                    onChange={(e) => setStudentIdDigits(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số dòng mỗi block MCQ:</label>
                                <input
                                    type="number"
                                    min={1}
                                    value={rowsPerBlock}
                                    onChange={(e) => setRowsPerBlock(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Tải file đáp án (.doc/.docx/.pdf/.txt):</label>
                                <input
                                    type="file"
                                    accept=".doc,.docx,.pdf,.txt"
                                    onChange={(e) => setTemplateAnswerKeyFile(e.target.files?.[0] || null)}
                                    style={inputStyle}
                                />
                                {templateAnswerKeyFile && (
                                    <div style={{ marginTop: "6px", fontSize: "13px", color: "#2e7d32" }}>
                                        Đã chọn: {templateAnswerKeyFile.name}
                                    </div>
                                )}
                            </div>
                            <div style={{ marginBottom: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Chuỗi đáp án đúng (nếu không dùng file):</label>
                                <input
                                    type="text"
                                    value={templateAnswerKey}
                                    onChange={(e) => setTemplateAnswerKey(e.target.value)}
                                    placeholder="Ví dụ: 1,2,3,4,2,1,1,4"
                                    style={inputStyle}
                                />
                                <small style={{ color: "gray" }}>
                                    Quy ước: A=1, B=2, C=3, D=4, E=5.
                                </small>
                            </div>
                            <button
                                className="submit-btn"
                                onClick={handleCreateTemplate}
                                disabled={isGeneratingTemplate}
                                style={{ width: "100%", opacity: isGeneratingTemplate ? 0.7 : 1 }}
                            >
                                {isGeneratingTemplate ? "Đang tạo phiếu..." : "Tạo phiếu"}
                            </button>
                            {templateDraft && (
                                <button
                                    className="submit-btn"
                                    onClick={handleSaveTemplate}
                                    disabled={isSavingTemplate}
                                    style={{ width: "100%", marginTop: "8px", background: "#2e7d32", opacity: isSavingTemplate ? 0.7 : 1 }}
                                >
                                    {isSavingTemplate ? "Đang lưu phiếu..." : "Lưu phiếu vào cơ sở dữ liệu"}
                                </button>
                            )}
                            {templateImageUrl && (
                                <div style={{ marginTop: "12px" }}>
                                    <p style={{ fontWeight: "bold", marginBottom: "8px" }}>Phiếu OMR đã tạo:</p>
                                    <img
                                        src={templateImageUrl}
                                        alt="Generated OMR Template"
                                        style={{ maxWidth: "100%", border: "2px solid #ddd", borderRadius: "4px", cursor: "pointer" }}
                                        onClick={() => setViewImage(templateImageUrl)}
                                    />
                                    <div style={{ marginTop: "8px" }}>
                                        <button
                                            type="button"
                                            className="submit-btn"
                                            style={{ width: "180px", background: "#1565c0", padding: "6px" }}
                                            onClick={() => handleDownloadFile(templateImageUrl, `${examTitle || "omr_template"}.png`)}
                                        >
                                            Tải phiếu
                                        </button>
                                    </div>
                                    {templateDraft && (
                                        <div style={{ marginTop: "8px", fontSize: "13px", color: "#6d4c41" }}>
                                            Phiếu hiện đang ở trạng thái nháp. Nhấn "Lưu phiếu vào cơ sở dữ liệu" để đưa vào Kho phiếu.
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                        )}

                        {activeMode === "grading" && (
                        <div className="config-section" style={{ marginBottom: "20px", padding: "15px", border: "1px solid #ddd", borderRadius: "8px" }}>
                            <h3>2. Cấu hình chấm điểm</h3>
                            <div style={{ marginBottom: "10px", padding: "10px", borderRadius: "6px", background: "#f5f7fa", color: "#334" }}>
                                Hệ thống sẽ tự nhận diện <strong>Tên bài thi</strong> và <strong>Mã đề</strong> từ ảnh, sau đó đối chiếu cơ sở dữ liệu để lấy đáp án đúng.
                            </div>
                            <div>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số câu hỏi:</label>
                                <input 
                                    type="number" 
                                    min={1}
                                    value={numQuestions} 
                                    onChange={(e) => setNumQuestions(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginTop: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số lựa chọn mỗi câu:</label>
                                <input
                                    type="number"
                                    min={2}
                                    value={numChoices}
                                    onChange={(e) => setNumChoices(Math.max(2, Number(e.target.value) || 2))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginTop: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số dòng mỗi block MCQ:</label>
                                <input
                                    type="number"
                                    min={1}
                                    value={rowsPerBlock}
                                    onChange={(e) => setRowsPerBlock(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                            <div style={{ marginTop: "10px" }}>
                                <label style={{ display: "block", fontWeight: "bold" }}>Số chữ số SID:</label>
                                <input
                                    type="number"
                                    min={1}
                                    value={studentIdDigits}
                                    onChange={(e) => setStudentIdDigits(Math.max(1, Number(e.target.value) || 1))}
                                    style={inputStyle}
                                />
                            </div>
                        </div>
                        )}

                        {activeMode === "grading" && (
                        <div className="upload-section">
                            <h3>3. Tải ảnh phiếu trả lời (tối đa 50 ảnh/lần gửi)</h3>
                            <div className="upload-area">
                                {omrImagePreviews.length === 0 && (
                                    <ImageUploader onClick={() => setShowPopup(true)} />
                                )}

                                {omrImagePreviews.length > 0 && (
                                    <div
                                        style={{
                                            width: "100%",
                                            border: "1px solid #ddd",
                                            borderRadius: "6px",
                                            background: "#fff",
                                            maxHeight: "240px",
                                            overflowY: "auto",
                                            padding: "8px",
                                        }}
                                    >
                                        {omrFiles.map((f, idx) => (
                                            <div
                                                key={`${f.name}-${idx}`}
                                                style={{
                                                    display: "flex",
                                                    justifyContent: "space-between",
                                                    alignItems: "center",
                                                    padding: "6px 4px",
                                                    borderBottom: "1px dashed #eee",
                                                    fontSize: "13px",
                                                }}
                                            >
                                                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingRight: "8px" }}>
                                                    {idx + 1}. {f.name}
                                                </span>
                                                <button
                                                    type="button"
                                                    className="submit-btn"
                                                    style={{ width: "72px", background: "#c62828", padding: "6px" }}
                                                    onClick={() => removeImage(idx)}
                                                >
                                                    Xóa
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {omrImagePreviews.length > 0 && (
                                <div style={{ marginTop: "12px", border: "1px solid #ddd", borderRadius: "8px", padding: "10px", background: "#fafafa" }}>
                                    <label style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: "bold", marginBottom: "8px" }}>
                                        <input
                                            type="checkbox"
                                            checked={manualCropEnabled}
                                            onChange={(e) => setManualCropEnabled(e.target.checked)}
                                            disabled={isBatchMode}
                                        />
                                        Cắt vùng thủ công trước khi chấm (khuyến nghị khi ảnh có nhiều vật thể)
                                    </label>
                                    {isBatchMode && (
                                        <p style={{ margin: "0 0 8px 0", fontSize: "13px", color: "#b26a00" }}>
                                            Chế độ chấm nhiều ảnh đang bật: tạm khóa Manual Crop để đảm bảo xử lý đồng loạt cùng đáp án.
                                        </p>
                                    )}
                                    {manualCropEnabled && !isBatchMode && (
                                        <>
                                            <p style={{ margin: "0 0 8px 0", fontSize: "13px", color: "#555" }}>
                                                Vung cat duoc chon bang 4 diem (4 goc). Nhap nut duoi day de mo popup va keo tung goc cho khop to de.
                                            </p>
                                            <div style={{ marginTop: "8px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                                                <button
                                                    type="button"
                                                    className="submit-btn"
                                                    onClick={() => setShowCropModal(true)}
                                                    style={{ width: "260px", background: "#2e7d32" }}
                                                >
                                                    Chinh 4 goc vung cat
                                                </button>
                                                <button
                                                    type="button"
                                                    className="submit-btn"
                                                    onClick={() => {
                                                        setCropQuad(buildDefaultCropQuad());
                                                    }}
                                                    style={{ width: "180px", background: "#78909c" }}
                                                >
                                                    Dat lai mac dinh
                                                </button>
                                            </div>
                                            {isSuggestingCrop && (
                                                <div style={{ marginTop: "8px", fontSize: "13px", color: "#1565c0" }}>
                                                    He thong dang phan tich va de xuat vung cat tu dong...
                                                </div>
                                            )}
                                            <div style={{ marginTop: "8px", fontSize: "12px", color: "#666" }}>
                                                TL({cropQuad[0].x.toFixed(2)}, {cropQuad[0].y.toFixed(2)}) | TR({cropQuad[1].x.toFixed(2)}, {cropQuad[1].y.toFixed(2)}) | BR({cropQuad[2].x.toFixed(2)}, {cropQuad[2].y.toFixed(2)}) | BL({cropQuad[3].x.toFixed(2)}, {cropQuad[3].y.toFixed(2)})
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                        )}

                        {activeMode === "grading" && (
                        <div className="submit-area" style={{ marginTop: "20px" }}>
                            {errorMessage && (
                                <div style={{
                                    color: "#f44336",
                                    fontSize: "14px",
                                    padding: "10px",
                                    backgroundColor: "#ffebee",
                                    borderRadius: "4px",
                                    marginBottom: "10px"
                                }}>
                                    {errorMessage}
                                </div>
                            )}
                            <button
                                className="submit-btn"
                                onClick={handleSubmit}
                                disabled={isLoading}
                                style={{ 
                                    width: "100%", 
                                    opacity: isLoading ? 0.7 : 1,
                                    cursor: isLoading ? "not-allowed" : "pointer"
                                }}
                            >
                                {isLoading ? "Đang chấm điểm..." : isBatchMode ? `Chấm điểm ${omrFiles.length} ảnh` : "Chấm điểm ngay"}
                            </button>
                        </div>
                        )}

                        {activeMode === "vault" && (
                        <div className="config-section" style={{ marginBottom: "20px", padding: "15px", border: "1px solid #ddd", borderRadius: "8px" }}>
                            <h3>2. Kho phiếu OMR</h3>
                            <div style={{ fontSize: "13px", color: "#555", marginBottom: "10px" }}>
                                Quản lý các phiếu đã lưu: tải ảnh phiếu cũ hoặc xóa phiếu để dọn cơ sở dữ liệu.
                            </div>
                            <button
                                className="submit-btn"
                                onClick={loadVault}
                                style={{ width: "100%", background: "#6d4c41" }}
                            >
                                Làm mới Kho phiếu
                            </button>
                        </div>
                        )}

                        {activeMode === "template" && errorMessage && (
                            <div style={{
                                color: "#f44336",
                                fontSize: "14px",
                                padding: "10px",
                                backgroundColor: "#ffebee",
                                borderRadius: "4px",
                                marginTop: "10px"
                            }}>
                                {errorMessage}
                            </div>
                        )}
                    </div>

                    {/* CỘT PHẢI: KẾT QUẢ */}
                    <div style={{ flex: 1, minWidth: "300px", maxWidth: "600px" }}>
                        <h3>{activeMode === "template" ? "4. Xem trước phiếu" : activeMode === "grading" ? "4. Kết quả chấm" : "4. Danh sách phiếu đã lưu"}</h3>
                        {activeMode === "template" && (
                            <div style={{ padding: "20px", textAlign: "center", color: "gray", border: "1px dashed #ccc" }}>
                                Nhấn "Tạo phiếu" để xem và tải phiếu mới.
                            </div>
                        )}
                        {activeMode === "grading" && (
                        batchResults.length > 0 ? (
                            <div style={{ border: "1px solid #4CAF50", padding: "15px", borderRadius: "8px", backgroundColor: "#f9fff9" }}>
                                <h2 style={{ color: "#2e7d32", textAlign: "center" }}>
                                    Kết quả chấm batch: {batchResults.filter((x) => x.success).length}/{batchResults.length} ảnh thành công
                                </h2>
                                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", justifyContent: "center", marginBottom: "10px" }}>
                                    <button
                                        className="submit-btn"
                                        style={{ width: "110px", background: batchFilter === "all" ? "#1976d2" : "#78909c" }}
                                        onClick={() => setBatchFilter("all")}
                                    >
                                        Tất cả
                                    </button>
                                    <button
                                        className="submit-btn"
                                        style={{ width: "110px", background: batchFilter === "failed" ? "#c62828" : "#78909c" }}
                                        onClick={() => setBatchFilter("failed")}
                                    >
                                        Lỗi
                                    </button>
                                    {batchZipUrl && (
                                        <a href={batchZipUrl} target="_blank" rel="noreferrer" style={{ alignSelf: "center", fontWeight: "bold" }}>
                                            Tải ZIP kết quả
                                        </a>
                                    )}
                                </div>
                                <div
                                    style={{
                                        maxHeight: "460px",
                                        overflowY: "auto",
                                        border: "1px solid #ddd",
                                        borderRadius: "6px",
                                        background: "#fff",
                                        padding: "8px",
                                    }}
                                >
                                    {batchResults
                                    .filter((item) => {
                                        if (batchFilter === "all") return true;
                                        if (batchFilter === "failed") return !item.success;
                                        return true;
                                    })
                                    .map((item, idx) => {
                                        if (!item.success) {
                                            return (
                                                <div key={`${item.file_name}-${idx}`} style={{ padding: "8px", borderBottom: "1px dashed #eee", color: "#b71c1c" }}>
                                                    <strong>{item.file_name}</strong> - Loi: {item.error || "Khong the cham"}
                                                </div>
                                            );
                                        }
                                        const one = item.data;
                                        return (
                                            <div key={`${item.file_name}-${idx}`} style={{ padding: "10px", borderBottom: "1px dashed #eee" }}>
                                                <div style={{ fontWeight: "bold" }}>{item.file_name}</div>
                                                <div style={{ fontSize: "13px", marginTop: "4px" }}>
                                                    Điểm: <strong>{one?.score}</strong> | SID: <strong>{one?.student_id || "-"}</strong> | Mã đề: <strong>{one?.exam_code || "-"}</strong> | Warp: {one?.warp_strategy || "-"}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        ) : serverResult ? (
                            <div style={{ border: "1px solid #4CAF50", padding: "15px", borderRadius: "8px", backgroundColor: "#f9fff9" }}>
                                <h2 style={{ color: "#2e7d32", textAlign: "center", marginBottom: "10px" }}>
                                    Kết quả chấm
                                </h2>
                                <div style={{ border: "1px solid #ddd", borderRadius: "6px", background: "#fff", padding: "10px" }}>
                                    <div style={{ fontSize: "13px", borderBottom: "1px dashed #eee", paddingBottom: "6px", marginBottom: "6px" }}>
                                        <strong>{omrFiles[0]?.name || "Ảnh 1"}</strong>
                                    </div>
                                    <div style={{ fontSize: "13px" }}>
                                        Điểm: <strong>{serverResult.score}</strong> | SID: <strong>{serverResult.student_id || "-"}</strong> | Mã đề: <strong>{serverResult.exam_code || "-"}</strong> | Warp: {serverResult.warp_strategy || "-"}
                                    </div>
                                </div>
                                {resultImageUrl && (
                                    <div style={{ textAlign: "center", marginTop: "12px" }}>
                                        <p style={{ fontWeight: "bold" }}>Ảnh kết quả:</p>
                                        <img
                                            src={resultImageUrl}
                                            alt="Graded Result"
                                            style={{ maxWidth: "100%", border: "2px solid #ddd", borderRadius: "4px", cursor: "pointer" }}
                                            onClick={() => setViewImage(resultImageUrl)}
                                        />
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ padding: "20px", textAlign: "center", color: "gray", border: "1px dashed #ccc" }}>
                                Chưa có kết quả. Vui lòng tải ảnh và nhấn Chấm điểm.
                            </div>
                        )
                        )}
                        {activeMode === "vault" && (
                            <div style={{ border: "1px solid #ddd", borderRadius: "8px", background: "#fff", padding: "10px" }}>
                                {isLoadingVault ? (
                                    <div style={{ padding: "12px", color: "#666" }}>Đang tải dữ liệu kho phiếu...</div>
                                ) : vaultItems.length === 0 ? (
                                    <div style={{ padding: "12px", color: "#666" }}>Kho phiếu đang trống.</div>
                                ) : (
                                    <div style={{ maxHeight: "520px", overflowY: "auto" }}>
                                        {vaultItems.map((item) => (
                                            <div key={item.omrid} style={{ borderBottom: "1px dashed #eee", padding: "10px 0" }}>
                                                <div style={{ fontWeight: "bold" }}>[{item.omr_code}] {item.omr_name}</div>
                                                <div style={{ fontSize: "13px", marginTop: "4px" }}>
                                                    Số câu: {item.omr_quest} | Tạo lúc: {item.created_at ? new Date(item.created_at).toLocaleString("vi-VN") : "-"}
                                                </div>
                                                <div style={{ fontSize: "12px", color: "#555", marginTop: "4px" }}>
                                                    Đáp án: {item.answer_preview || "-"}
                                                </div>
                                                <div style={{ marginTop: "8px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                                                    {item.template_url ? (
                                                        <button
                                                            type="button"
                                                            className="submit-btn"
                                                            style={{ width: "130px", background: "#1565c0", padding: "6px" }}
                                                            onClick={() => handleDownloadFile(`${API_CONFIG.BASE_URL}${item.template_url}`, `${item.omr_name || "omr"}_${item.omr_code || ""}.png`)}
                                                        >
                                                            Tải ảnh phiếu
                                                        </button>
                                                    ) : (
                                                        <span style={{ color: "#999" }}>Phiếu này chưa có ảnh lưu</span>
                                                    )}
                                                    <button
                                                        type="button"
                                                        className="submit-btn"
                                                        style={{ width: "120px", background: "#c62828", padding: "6px" }}
                                                        onClick={() => handleDeleteVaultItem(Number(item.omrid))}
                                                    >
                                                        Xóa phiếu
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Popup chọn nguồn ảnh */}
            {showPopup && (
                <UploadPopup
                    onSelect={(files: FileList | null) => {
                        handleImages(files);
                        setShowPopup(false);
                    }}
                    onClose={() => setShowPopup(false)}
                />
            )}

            {/* Popup xem ảnh lớn */}
            {viewImage && (
                <ViewImageModal
                    img={viewImage}
                    onClose={() => setViewImage(null)}
                />
            )}

            {/* Popup chinh 4 goc crop OMR */}
            {showCropModal && omrImagePreviews[0] && !isBatchMode && (
                <OmrQuadCropModal
                    imageUrl={omrImagePreviews[0]}
                    initialPoints={cropQuad}
                    onCancel={() => setShowCropModal(false)}
                    onConfirm={(points) => {
                        setCropQuad(points);
                        setShowCropModal(false);
                    }}
                />
            )}
        </div>
    );
}