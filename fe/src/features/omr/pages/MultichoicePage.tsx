import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { API_CONFIG } from "../../../config/api";
import ViewImageModal from "../../../components/ViewImageModal";
import { getAuthUid } from "../../../utils/authStorage";
import ExportPanel from "../components/ExportPanel";
import StatsPanel from "../components/StatsPanel";
import "../styles/OmrMobileApp.css";
import {
  AUTO_CAPTURE_COOLDOWN_MS,
  CORNER_KEYS,
  FALLBACK_MARKER_SAMPLES,
  OPTION_LABELS,
} from "../types";
import type {
  AnswerSet,
  AssignmentApiItem,
  BatchGradeResultItem,
  CornerKey,
  DetailTab,
  FormProfile,
  MarkerSample,
  NavTab,
  OMRResult,
  PickedImageSource,
  ScannerUiState,
  TestCardItem,
} from "../types";
import {
  avgLumaInArea,
  buildGradeRecord,
  clampNumber,
  darkRatioInArea,
  formatDateTimeLabel,
  getGradeRecordsFromLastResult,
  getHandwritingOverlayRois,
  mapClientPointToVideo,
  mergeLastResultWithRecords,
  nowDateTimeLocal,
  rebuildLastResultFromRecords,
  toAbsoluteStaticUrl,
  toAnswerString,
  toStaticOmrUrlFromFileName,
} from "../utils";

export default function MultichoicePage() {
  const navigate = useNavigate();
  const { testId: routeTestIdRaw, recordId: routeRecordIdRaw } = useParams();
  const uid = getAuthUid();

  const routeTestId = useMemo(() => {
    if (!routeTestIdRaw) return null;
    const parsed = Number(routeTestIdRaw);
    if (!Number.isInteger(parsed) || parsed <= 0) return null;
    return parsed;
  }, [routeTestIdRaw]);

  const routeRecordId = useMemo(() => {
    if (!routeRecordIdRaw) return null;
    try {
      return decodeURIComponent(routeRecordIdRaw);
    } catch {
      return routeRecordIdRaw;
    }
  }, [routeRecordIdRaw]);

  const isRecordDetailPage = routeTestId !== null;

  const [navTab, setNavTab] = useState<NavTab>("home");
  const [detailTestId, setDetailTestId] = useState<number | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("grading");

  const [formProfiles, setFormProfiles] = useState<FormProfile[]>([]);
  const [formProfilesLoading, setFormProfilesLoading] = useState(false);

  const [tests, setTests] = useState<TestCardItem[]>([]);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newFormProfileCode, setNewFormProfileCode] = useState("");
  const [newCreatedAt, setNewCreatedAt] = useState(nowDateTimeLocal());

  const [newCode, setNewCode] = useState("");
  const [statsFilterMode, setStatsFilterMode] = useState<"exam_code" | "student_id">("exam_code");
  const [statsFilterValue, setStatsFilterValue] = useState("");

  const [pickedFiles, setPickedFiles] = useState<File[]>([]);
  const [pickedPreviews, setPickedPreviews] = useState<string[]>([]);
  const [pickedSource, setPickedSource] = useState<PickedImageSource>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [batchSummary, setBatchSummary] = useState<{
    successCount: number;
    failedCount: number;
    totalCount: number;
    zipUrl: string | null;
  } | null>(null);
  const [viewImage, setViewImage] = useState<string | null>(null);
  const [selectedGradeRecordId, setSelectedGradeRecordId] = useState<string | null>(null);
  const [samplePreview, setSamplePreview] = useState<{ title: string; fileName: string; url: string } | null>(null);

  const libraryInputRef = useRef<HTMLInputElement>(null);
  const previewUrlsRef = useRef<Set<string>>(new Set());

  const cameraStageRef = useRef<HTMLDivElement>(null);
  const viewfinderRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectRafRef = useRef<number | null>(null);
  const detectCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectLastTsRef = useRef(0);
  const detectHitRef = useRef(0);
  const detectMissRef = useRef(0);
  const autoCaptureLastTsRef = useRef(0);
  const autoCaptureBusyRef = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [scannerState, setScannerState] = useState<ScannerUiState>("idle");

  const selectedTest = useMemo(() => {
    if (routeTestId !== null) {
      return tests.find((x) => x.id === routeTestId) || null;
    }
    return tests.find((x) => x.id === detailTestId) || null;
  }, [tests, detailTestId, routeTestId]);

  const selectedNewProfile = useMemo(
    () => formProfiles.find((p) => p.code === newFormProfileCode) || null,
    [formProfiles, newFormProfileCode]
  );

  const selectedTestProfile = useMemo(() => {
    if (!selectedTest?.formProfileCode) return null;
    return formProfiles.find((p) => p.code === selectedTest.formProfileCode) || null;
  }, [formProfiles, selectedTest]);

  const scannerAspectRatio = useMemo(() => {
    const raw = Number(selectedTestProfile?.strategy?.sheet_aspect_ratio ?? 1.414214);
    return clampNumber(raw, 1.25, 1.8);
  }, [selectedTestProfile]);

  const scannerHint = useMemo(() => {
    const hint = selectedTestProfile?.strategy?.scanner_hint;
    return {
      minDarkRatio: clampNumber(Number(hint?.min_dark_ratio ?? 0.14), 0.08, 0.35),
      minCenterLuma: clampNumber(Number(hint?.min_center_luma ?? 52), 35, 120),
      minMarkerContrast: clampNumber(Number(hint?.min_marker_contrast ?? 20), 10, 70),
      sampleSizeNorm: clampNumber(Number(hint?.sample_size_norm ?? 0.062), 0.028, 0.14),
    };
  }, [selectedTestProfile]);

  const markerSamples = useMemo<Record<CornerKey, MarkerSample>>(() => {
    const markerMap = selectedTestProfile?.strategy?.corner_markers;
    const next: Record<CornerKey, MarkerSample> = {
      tl: { ...FALLBACK_MARKER_SAMPLES.tl },
      tr: { ...FALLBACK_MARKER_SAMPLES.tr },
      bl: { ...FALLBACK_MARKER_SAMPLES.bl },
      br: { ...FALLBACK_MARKER_SAMPLES.br },
    };

    for (const key of CORNER_KEYS) {
      const marker = markerMap?.[key];
      if (!marker) continue;

      const centerX = Number(marker.cx ?? (marker.x + (marker.w / 2)));
      const centerY = Number(marker.cy ?? (marker.y + (marker.h / 2)));
      const inferredSize = Number(Math.max(marker.w, marker.h) * 1.8);

      next[key] = {
        x: clampNumber(centerX, 0.04, 0.96),
        y: clampNumber(centerY, 0.04, 0.96),
        sampleSize: clampNumber(
          Number.isFinite(inferredSize) && inferredSize > 0 ? inferredSize : scannerHint.sampleSizeNorm,
          0.028,
          0.14
        ),
      };
    }

    return next;
  }, [selectedTestProfile, scannerHint.sampleSizeNorm]);

  const showDetail = !isRecordDetailPage && !!selectedTest;

  const selectedAnswerSet = useMemo(() => {
    if (!selectedTest || !selectedTest.activeCode) return null;
    return selectedTest.answerSets.find((x) => x.code === selectedTest.activeCode) || null;
  }, [selectedTest]);

  const selectedAnswerCount = useMemo(() => {
    if (!selectedAnswerSet) return 0;
    return selectedAnswerSet.answers.filter((x) => !!x).length;
  }, [selectedAnswerSet]);

  const gradeRecords = useMemo(
    () => getGradeRecordsFromLastResult(selectedTest?.lastResult),
    [selectedTest?.lastResult]
  );

  const normalizedStatsFilterValue = useMemo(() => statsFilterValue.trim().toLowerCase(), [statsFilterValue]);

  const filteredGradeRecords = useMemo(() => {
    if (!normalizedStatsFilterValue) return gradeRecords;

    return gradeRecords.filter((record) => {
      const candidate = statsFilterMode === "exam_code"
        ? String(record.data?.exam_code ?? "")
        : String(record.data?.student_id ?? "");
      return candidate.toLowerCase().includes(normalizedStatsFilterValue);
    });
  }, [gradeRecords, normalizedStatsFilterValue, statsFilterMode]);

  const selectedGradeRecord = useMemo(() => {
    if (!gradeRecords.length) return null;
    if (!selectedGradeRecordId) return gradeRecords[0];
    return gradeRecords.find((record) => record.id === selectedGradeRecordId) || gradeRecords[0];
  }, [gradeRecords, selectedGradeRecordId]);

  const handwritingOverlayRois = useMemo(() => {
    return getHandwritingOverlayRois(selectedGradeRecord?.data || null);
  }, [selectedGradeRecord]);

  const gradeBlockReason = useMemo(() => {
    if (!selectedTest) return "Chưa chọn bài kiểm tra.";
    if (!selectedAnswerSet) return "Vui lòng thêm mã đề và chọn bộ đáp án đang dùng.";
    if (selectedAnswerCount <= 0) return "Vui lòng điền ít nhất 1 đáp án.";
    if (!pickedFiles.length) return "Vui lòng thêm ảnh bài làm trước khi chấm.";
    return "";
  }, [selectedTest, selectedAnswerSet, selectedAnswerCount, pickedFiles.length]);

  const canGradeNow = !submitting && !gradeBlockReason;

  const mapAssignmentToCard = useCallback((item: AssignmentApiItem): TestCardItem => {
    return {
      id: Number(item.aid),
      title: String(item.title || "Bài thi"),
      createdAt: String(item.created_at_label || ""),
      createdAtRaw: String(item.created_at_raw || ""),
      questionCount: Math.max(1, Number(item.question_count || 1)),
      totalPoints: Math.max(1, Number(item.total_points || 1)),
      formProfileCode: item.form_profile_code || null,
      gradedCount: Math.max(0, Number(item.graded_count || 0)),
      answerSets: Array.isArray(item.answer_sets) ? item.answer_sets : [],
      activeCode: item.active_code || null,
      lastResult: item.last_result,
    };
  }, []);

  const loadFormProfiles = useCallback(async () => {
    setFormProfilesLoading(true);
    try {
      const res = await fetch(API_CONFIG.OMR.LIST_FORM_PROFILES);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Không tải được profile phiếu mẫu");
      const rows: FormProfile[] = Array.isArray(data.profiles) ? data.profiles : [];
      setFormProfiles(rows);
      setNewFormProfileCode((prev) => {
        if (prev) return prev;
        if (rows.length <= 0) return prev;
        return String(rows[0].code || "");
      });
    } catch (err) {
      console.error(err);
      setErrorMessage(err instanceof Error ? err.message : "Không tải được profile phiếu mẫu.");
    } finally {
      setFormProfilesLoading(false);
    }
  }, []);

  const loadAssignments = useCallback(async () => {
    if (!uid) {
      setTests([]);
      return;
    }
    try {
      const res = await fetch(API_CONFIG.OMR.LIST_ASSIGNMENTS(uid));
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Không tải được danh sách bài thi");
      const rows: AssignmentApiItem[] = Array.isArray(data.assignments) ? data.assignments : [];
      setTests(rows.map(mapAssignmentToCard));
    } catch (err) {
      console.error(err);
      setErrorMessage(err instanceof Error ? err.message : "Không tải được danh sách bài thi.");
    }
  }, [uid, mapAssignmentToCard]);

  useEffect(() => {
    void loadAssignments();
    void loadFormProfiles();
  }, [loadAssignments, loadFormProfiles]);

  useEffect(() => {
    const previewUrls = previewUrlsRef.current;
    const videoEl = videoRef.current;
    return () => {
      stopDetectionLoop();
      previewUrls.forEach((url) => URL.revokeObjectURL(url));
      previewUrls.clear();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!successMessage) return;
    const timer = window.setTimeout(() => setSuccessMessage(""), 2200);
    return () => window.clearTimeout(timer);
  }, [successMessage]);

  useEffect(() => {
    setStatsFilterMode("exam_code");
    setStatsFilterValue("");
  }, [selectedTest?.id]);

  useEffect(() => {
    if (gradeRecords.length === 0) {
      if (selectedGradeRecordId) {
        setSelectedGradeRecordId(null);
      }
      return;
    }

    if (isRecordDetailPage && routeRecordId) {
      if (gradeRecords.some((record) => record.id === routeRecordId)) {
        if (selectedGradeRecordId !== routeRecordId) {
          setSelectedGradeRecordId(routeRecordId);
        }
        return;
      }
    }

    if (!selectedGradeRecordId || !gradeRecords.some((record) => record.id === selectedGradeRecordId)) {
      setSelectedGradeRecordId(gradeRecords[0].id);
    }
  }, [gradeRecords, selectedGradeRecordId, isRecordDetailPage, routeRecordId]);

  useEffect(() => {
    if (!showDetail || detailTab !== "grading") {
      stopDetectionLoop();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      autoCaptureBusyRef.current = false;
      autoCaptureLastTsRef.current = 0;
      setCameraOn(false);
      setScannerState("idle");
      return;
    }
    // Keep tab entry passive to avoid immediate errors on mobile HTTP/LAN contexts.
    autoCaptureBusyRef.current = false;
    setScannerState("searching");
    setCameraError("");
  }, [showDetail, detailTab, detailTestId]);

  useEffect(() => {
    if (!cameraOn || !showDetail || detailTab !== "grading") {
      stopDetectionLoop();
      return;
    }

    const tick = (ts: number) => {
      if (!cameraOn || !videoRef.current) {
        stopDetectionLoop();
        return;
      }

      if (ts - detectLastTsRef.current > 130) {
        detectLastTsRef.current = ts;
        const matched = evaluateAlignment();
        if (matched) {
          detectHitRef.current += 1;
          detectMissRef.current = 0;
          if (detectHitRef.current >= 4 && scannerState !== "locked") {
            setScannerState("locked");
          }
        } else {
          detectMissRef.current += 1;
          detectHitRef.current = 0;
          if (detectMissRef.current >= 4 && scannerState !== "searching") {
            setScannerState("searching");
          }
        }
      }

      detectRafRef.current = requestAnimationFrame(tick);
    };

    stopDetectionLoop();
    detectHitRef.current = 0;
    detectMissRef.current = 0;
    detectLastTsRef.current = 0;
    detectRafRef.current = requestAnimationFrame(tick);

    return () => stopDetectionLoop();
  }, [cameraOn, showDetail, detailTab, scannerState, evaluateAlignment]);

  useEffect(() => {
    if (!cameraOn || !showDetail || detailTab !== "grading" || scannerState !== "locked") {
      return;
    }

    const now = Date.now();
    if (autoCaptureBusyRef.current) return;
    if (now - autoCaptureLastTsRef.current < AUTO_CAPTURE_COOLDOWN_MS) return;

    autoCaptureBusyRef.current = true;
    void captureCurrentFrame().finally(() => {
      autoCaptureBusyRef.current = false;
      autoCaptureLastTsRef.current = Date.now();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOn, showDetail, detailTab, scannerState]);

  useEffect(() => {
    if (!cameraOn || !videoRef.current || !streamRef.current) return;
    if (videoRef.current.srcObject !== streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
    void videoRef.current.play().catch(() => undefined);
  }, [cameraOn]);

  function stopDetectionLoop() {
    if (detectRafRef.current !== null) {
      cancelAnimationFrame(detectRafRef.current);
      detectRafRef.current = null;
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  function evaluateAlignment() {
    const videoEl = videoRef.current;
    const vfEl = viewfinderRef.current;
    if (!videoEl || !vfEl || videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) {
      return false;
    }

    if (!detectCanvasRef.current) {
      detectCanvasRef.current = document.createElement("canvas");
    }
    const canvas = detectCanvasRef.current;
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return false;

    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    const videoRect = videoEl.getBoundingClientRect();
    const vfRect = vfEl.getBoundingClientRect();

    const center = mapClientPointToVideo(vfRect.left + vfRect.width / 2, vfRect.top + vfRect.height / 2, videoEl, videoRect);
    const centerLuma = avgLumaInArea(
      ctx,
      center.x - (videoEl.videoWidth * 0.07),
      center.y - (videoEl.videoHeight * 0.07),
      videoEl.videoWidth * 0.14,
      videoEl.videoHeight * 0.14
    );

    const darkLumaThreshold = clampNumber(centerLuma - 26, 28, 110);
    const markerStats = CORNER_KEYS.map((corner) => {
      const sample = markerSamples[corner];
      const point = mapClientPointToVideo(
        vfRect.left + (sample.x * vfRect.width),
        vfRect.top + (sample.y * vfRect.height),
        videoEl,
        videoRect
      );

      const samplePx = Math.max(
        12,
        Math.min(videoEl.videoWidth, videoEl.videoHeight) * sample.sampleSize
      );

      return {
        darkRatio: darkRatioInArea(ctx, point.x, point.y, samplePx, darkLumaThreshold),
        markerLuma: avgLumaInArea(ctx, point.x - (samplePx / 2), point.y - (samplePx / 2), samplePx, samplePx),
      };
    });

    const markerLumaAvg = markerStats.reduce((sum, item) => sum + item.markerLuma, 0) / markerStats.length;

    const hasFourDarkMarkers = markerStats.every((item) => item.darkRatio >= scannerHint.minDarkRatio);
    const paperInsideFrame = centerLuma >= scannerHint.minCenterLuma;
    const markerContrastOk = (centerLuma - markerLumaAvg) >= scannerHint.minMarkerContrast;

    return hasFourDarkMarkers && paperInsideFrame && markerContrastOk;
  }

  function clearPickedMedia() {
    setPickedPreviews((prev) => {
      prev.forEach((url) => {
        if (previewUrlsRef.current.has(url)) {
          URL.revokeObjectURL(url);
          previewUrlsRef.current.delete(url);
        }
      });
      return [];
    });
    setPickedFiles([]);
    setPickedSource(null);
  }

  function openTestForGrading(testId: number) {
    setDetailTestId(testId);
    setDetailTab("grading");
    setNavTab("home");
    setErrorMessage("");
    setSuccessMessage("");
    setBatchSummary(null);
    setResultImageUrl(null);
    setViewImage(null);
    setSelectedGradeRecordId(null);
    clearPickedMedia();
    stopCamera();
    navigate("/multichoice");
  }

  async function deleteAssignment(aid: number) {
    if (!uid) return;
    if (!window.confirm("Bạn chắc chắn muốn xóa bài thi này?")) return;
    try {
      const res = await fetch(API_CONFIG.OMR.DELETE_ASSIGNMENT(uid, aid), { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.message || "Xóa bài thi thất bại");
      setTests((prev) => prev.filter((x) => x.id !== aid));
      if (detailTestId === aid) {
        setDetailTestId(null);
        setNavTab("home");
      }
      setSuccessMessage("Đã xóa bài thi.");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Xóa bài thi thất bại");
    }
  }

  function openRecordDetail(record: { id: string }) {
    if (!selectedTest) return;
    setSelectedGradeRecordId(record.id);
    stopCamera();
    navigate(`/multichoice/record-detail/${selectedTest.id}/${encodeURIComponent(record.id)}`);
  }

  function deleteGradeRecord(recordId: string) {
    if (!selectedTest) return;
    const currentRecords = getGradeRecordsFromLastResult(selectedTest.lastResult);
    const target = currentRecords.find((record) => record.id === recordId);
    if (!target) return;

    if (!window.confirm(`Xóa bản ghi đã chấm cho ảnh ${target.file_name}?`)) return;

    const nextRecords = currentRecords.filter((record) => record.id !== recordId);
    const nextLastResult = rebuildLastResultFromRecords(selectedTest.lastResult, nextRecords);

    setSelectedGradeRecordId(nextRecords[0]?.id || null);
    setResultImageUrl(nextRecords[0]?.image_url || null);
    updateSelectedTest({
      gradedCount: nextRecords.length,
      lastResult: nextLastResult,
    });
    setSuccessMessage("Đã xóa 1 bản ghi khỏi lịch sử chấm.");
  }

  async function persistAssignment(item: TestCardItem) {
    if (!uid || !item?.id) return;
    const formData = new FormData();
    formData.append("title", item.title);
    formData.append("created_at_raw", item.createdAtRaw || "");
    formData.append("created_at_label", item.createdAt || "");
    formData.append("question_count", String(item.questionCount));
    formData.append("total_points", String(item.totalPoints));
    formData.append("form_profile_code", item.formProfileCode || "");
    formData.append("graded_count", String(item.gradedCount));
    formData.append("answer_sets", JSON.stringify(item.answerSets || []));
    formData.append("active_code", item.activeCode || "");
    if (item.lastResult) {
      formData.append("last_result", JSON.stringify(item.lastResult));
    }
    const res = await fetch(API_CONFIG.OMR.UPDATE_ASSIGNMENT(uid, item.id), {
      method: "PUT",
      body: formData,
    });
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || payload.message || "Lưu bài thi thất bại");
    }
  }

  function updateSelectedTest(next: Partial<TestCardItem>) {
    if (!selectedTest) return;
    const merged = { ...selectedTest, ...next };
    setTests((prev) => prev.map((t) => (t.id === selectedTest.id ? merged : t)));
    void persistAssignment(merged).catch((err) => {
      setErrorMessage(err instanceof Error ? err.message : "Không thể lưu dữ liệu bài thi.");
    });
  }

  function onPickFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    const room = Math.max(0, 50 - pickedFiles.length);
    const accepted = files.slice(0, room);
    if (accepted.length === 0) {
        setErrorMessage("Tối đa 50 ảnh cho một lần chấm.");
        return;
    }
    const acceptedPreviews = accepted.map((f) => URL.createObjectURL(f));
    acceptedPreviews.forEach((url) => previewUrlsRef.current.add(url));
    setPickedFiles((prev) => [...prev, ...accepted]);
    setPickedPreviews((prev) => [...prev, ...acceptedPreviews]);
    setPickedSource((prev) => (prev && prev !== "library" ? "mixed" : "library"));
    if (accepted.length < files.length) {
      setErrorMessage("Đã đạt giới hạn 50 ảnh, một số ảnh chưa được thêm.");
      return;
    }
    setErrorMessage("");
  }

  function removePickedFile(idx: number) {
    setPickedPreviews((prev) => {
      const target = prev[idx];
      if (target && previewUrlsRef.current.has(target)) {
        URL.revokeObjectURL(target);
        previewUrlsRef.current.delete(target);
      }
      return prev.filter((_, i) => i !== idx);
    });
    setPickedFiles((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      if (next.length === 0) {
        setPickedSource(null);
      }
      return next;
    });
  }

  async function captureCurrentFrame() {
    const videoEl = videoRef.current;
    if (!videoEl || !videoEl.videoWidth || !videoEl.videoHeight) {
      setErrorMessage("Camera chưa sẵn sàng để chụp.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setErrorMessage("Không tạo được vùng chụp ảnh.");
      return;
    }
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92);
    });
    if (!blob) {
      setErrorMessage("Chụp ảnh thất bại. Vui lòng thử lại.");
      return;
    }

    const file = new File([blob], `camera_${Date.now()}.jpg`, { type: "image/jpeg" });
    const previewUrl = URL.createObjectURL(file);
    if (pickedFiles.length >= 50) {
      URL.revokeObjectURL(previewUrl);
      setErrorMessage("Tối đa 50 ảnh cho một lần chấm.");
      return;
    }
    previewUrlsRef.current.add(previewUrl);
    setPickedFiles((prev) => [...prev, file]);
    setPickedPreviews((prev) => [...prev, previewUrl]);
    setPickedSource((prev) => (prev && prev !== "camera" ? "mixed" : "camera"));
    setErrorMessage("");
  }

  async function startCamera() {
    setCameraError("");
    stopDetectionLoop();
    if (streamRef.current) {
      stopCamera();
      await new Promise((resolve) => window.setTimeout(resolve, 120));
    }

    // Keep scanner inline only; do not open native fullscreen capture as fallback.
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setCameraError("Thiết bị đang chặn camera live (thường do HTTP/LAN). Hãy dùng Tải ảnh từ thư viện.");
      return false;
    }

    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { exact: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
      setScannerState("searching");
      return true;
    } catch (err) {
      console.error(err);
      setCameraError("Không mở được camera live. Vui lòng cấp quyền camera hoặc dùng Tải ảnh từ thư viện.");
      setCameraOn(false);
      return false;
    }
  }

  function stopCamera() {
    stopDetectionLoop();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    autoCaptureBusyRef.current = false;
    autoCaptureLastTsRef.current = 0;
    setCameraOn(false);
  }

  async function startSmartScannerFlow() {
    autoCaptureBusyRef.current = false;
    autoCaptureLastTsRef.current = 0;
    setScannerState("searching");

    const canUseLiveCamera = await startCamera();
    if (!canUseLiveCamera) {
      setScannerState("idle");
      return false;
    }
    return true;
  }

  async function createTestFromSheet() {
    if (!newTitle.trim()) {
      setErrorMessage("Vui lòng nhập tên bài kiểm tra.");
      return;
    }

    if (!uid) {
      setErrorMessage("Bạn chưa đăng nhập.");
      return;
    }

    if (!selectedNewProfile) {
      setErrorMessage("Vui lòng chọn profile phiếu mẫu.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("uid", String(uid));
      formData.append("title", newTitle.trim());
      formData.append("created_at_raw", newCreatedAt);
      formData.append("created_at_label", formatDateTimeLabel(newCreatedAt));
      formData.append("question_count", String(Math.max(1, selectedNewProfile.default_questions)));
      formData.append("total_points", String(10));
      formData.append("form_profile_code", selectedNewProfile.code);

      const res = await fetch(API_CONFIG.OMR.CREATE_ASSIGNMENT, {
        method: "POST",
        body: formData,
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || "Tạo bài thi thất bại");
      }

      const created = mapAssignmentToCard(payload.assignment as AssignmentApiItem);
      setTests((prev) => [created, ...prev]);
      setSheetOpen(false);
      setNavTab("home");
      setErrorMessage("");
      setNewTitle("");
      setNewCreatedAt(nowDateTimeLocal());
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Tạo bài thi thất bại.");
    }
  }

  function addAnswerCode() {
    if (!selectedTest) return;
    const code = newCode.trim();
    if (!code) {
      setErrorMessage("Vui lòng nhập mã đề trước khi thêm.");
      return;
    }
    if (!/^\d{3}$/.test(code)) {
      setErrorMessage("Mã đề phải gồm đúng 3 chữ số (VD: 001).");
      return;
    }
    if (selectedTest.answerSets.some((x) => x.code === code)) {
      setErrorMessage("Mã đề đã tồn tại trong bài kiểm tra này.");
      return;
    }
    const nextSet: AnswerSet = {
      code,
      answers: Array.from({ length: selectedTest.questionCount }, () => ""),
    };
    updateSelectedTest({
      answerSets: [...selectedTest.answerSets, nextSet],
      activeCode: code,
    });
    setNewCode("");
    setErrorMessage("");
  }

  function removeAnswerCode(code: string) {
    if (!selectedTest) return;
    const next = selectedTest.answerSets.filter((x) => x.code !== code);
    const nextActive = selectedTest.activeCode === code ? (next[0]?.code || null) : selectedTest.activeCode;
    updateSelectedTest({ answerSets: next, activeCode: nextActive });
  }

  function updateAnswer(questionIdx: number, value: string) {
    if (!selectedTest || !selectedAnswerSet) return;
    const nextSets = selectedTest.answerSets.map((set) => {
      if (set.code !== selectedAnswerSet.code) return set;
      const answers = [...set.answers];
      answers[questionIdx] = answers[questionIdx] === value ? "" : value;
      return { ...set, answers };
    });
    updateSelectedTest({ answerSets: nextSets });
  }

  async function gradeCurrentTest() {
    if (!uid) {
      setErrorMessage("Bạn chưa đăng nhập.");
      return;
    }
    if (!selectedTest) {
      setErrorMessage("Chưa chọn bài kiểm tra.");
      return;
    }
    if (!selectedAnswerSet) {
      setErrorMessage("Vui lòng thêm mã đề và điền đáp án trước khi chấm.");
      return;
    }
    if (selectedAnswerCount <= 0) {
      setErrorMessage("Vui lòng điền ít nhất 1 đáp án trước khi chấm.");
      return;
    }
    if (!pickedFiles.length) {
      setErrorMessage("Vui lòng thêm ảnh bài làm.");
      return;
    }

    setSubmitting(true);
    setErrorMessage("");
    try {
      const runtimeNumChoices = selectedTestProfile?.num_choices || 4;
      const runtimeRowsPerBlock = selectedTestProfile?.rows_per_block || 20;
      const runtimeStudentDigits = selectedTestProfile?.student_id_digits || 6;
      const runtimeSidWrite = selectedTestProfile?.sid_has_write_row ?? true;
      const runtimeNumBlocks = selectedTestProfile?.num_blocks;
      const profileCode = selectedTest.formProfileCode || "";

      if (pickedFiles.length > 1) {
        const formData = new FormData();
        pickedFiles.forEach((f) => formData.append("files", f));
        formData.append("uid", String(uid));
        formData.append("aid", String(selectedTest.id));
        formData.append("answers", toAnswerString(selectedAnswerSet.answers));
        formData.append("num_questions", String(selectedTest.questionCount));
        formData.append("num_choices", String(runtimeNumChoices));
        formData.append("student_id_digits", String(runtimeStudentDigits));
        formData.append("rows_per_block", String(runtimeRowsPerBlock));
        formData.append("sid_has_write_row", String(runtimeSidWrite));
        if (runtimeNumBlocks != null) {
          formData.append("num_blocks", String(runtimeNumBlocks));
        }
        if (profileCode) {
          formData.append("form_profile_code", profileCode);
        }

        const res = await fetch(API_CONFIG.OMR.GRADE_BATCH, {
          method: "POST",
          body: formData,
        });
        const payload = await res.json();
        if (!res.ok) {
          throw new Error(payload.detail || payload.message || "Chấm batch thất bại");
        }

        const successItems: BatchGradeResultItem[] = Array.isArray(payload.results)
          ? payload.results.filter((x: BatchGradeResultItem) => x?.success && !!x?.data)
          : [];
        if (successItems.length === 0) {
          throw new Error("Không có ảnh nào chấm thành công trong lô batch.");
        }

        const newRecords = successItems.map((item, index) => {
          const itemData = item.data as OMRResult;
          const bubbleConfidenceJsonUrl =
            toAbsoluteStaticUrl(item.bubble_confidence_json_url)
            || toAbsoluteStaticUrl(itemData?.bubble_confidence_json_url)
            || toStaticOmrUrlFromFileName(itemData?.bubble_confidence_json);
          return buildGradeRecord({
            source: "batch",
            fileName: item.file_name || `batch_${index + 1}.jpg`,
            imageUrl: toAbsoluteStaticUrl(item.image_url),
            sidCropUrl: toAbsoluteStaticUrl(item.sid_crop_url),
            mcqCropUrl: toAbsoluteStaticUrl(item.mcq_crop_url),
            bubbleConfidenceJsonUrl,
            data: itemData,
          });
        }).reverse();

        const latestRecord = newRecords[0];
        const successCount = Number(payload.success_count || successItems.length);
        setResultImageUrl(latestRecord?.image_url || null);
        setSelectedGradeRecordId(latestRecord?.id || null);
        updateSelectedTest({
          gradedCount: Number(selectedTest.gradedCount || 0) + successCount,
          lastResult: mergeLastResultWithRecords(
            selectedTest.lastResult,
            latestRecord?.data || (successItems[successItems.length - 1].data as OMRResult),
            newRecords
          ),
        });
        const zipUrl = toAbsoluteStaticUrl(payload.zip_url) ?? null;
        const failedCount = Number(payload.failed_count ?? 0);
        clearPickedMedia();
        setBatchSummary({ successCount, failedCount, totalCount: pickedFiles.length, zipUrl });
        setDetailTab("stats");
        setSuccessMessage(`Đã chấm batch ${successCount}/${pickedFiles.length} ảnh.`);
      } else {
        const formData = new FormData();
        formData.append("file", pickedFiles[pickedFiles.length - 1]);
        formData.append("uid", String(uid));
        formData.append("aid", String(selectedTest.id));
        formData.append("answers", toAnswerString(selectedAnswerSet.answers));
        formData.append("num_questions", String(selectedTest.questionCount));
        formData.append("num_choices", String(runtimeNumChoices));
        formData.append("student_id_digits", String(runtimeStudentDigits));
        formData.append("rows_per_block", String(runtimeRowsPerBlock));
        formData.append("sid_has_write_row", String(runtimeSidWrite));
        if (runtimeNumBlocks != null) {
          formData.append("num_blocks", String(runtimeNumBlocks));
        }
        if (profileCode) {
          formData.append("form_profile_code", profileCode);
        }

        const res = await fetch(API_CONFIG.OMR.GRADE, {
          method: "POST",
          body: formData,
        });
        const payload = await res.json();
        if (!res.ok) {
          throw new Error(payload.detail || payload.message || "Chấm điểm thất bại");
        }

        const data = payload.data as OMRResult;
        const bubbleConfidenceJsonUrl =
          toAbsoluteStaticUrl(payload.bubble_confidence_json_url)
          || toAbsoluteStaticUrl(data?.bubble_confidence_json_url)
          || toStaticOmrUrlFromFileName(data?.bubble_confidence_json);
        const singleRecord = buildGradeRecord({
          source: "single",
          fileName: pickedFiles[pickedFiles.length - 1]?.name || `camera_${Date.now()}.jpg`,
          imageUrl: toAbsoluteStaticUrl(payload.image_url),
          sidCropUrl: toAbsoluteStaticUrl(payload.sid_crop_url),
          mcqCropUrl: toAbsoluteStaticUrl(payload.mcq_crop_url),
          bubbleConfidenceJsonUrl,
          data,
        });

        setResultImageUrl(singleRecord.image_url || null);
        setSelectedGradeRecordId(singleRecord.id);
        updateSelectedTest({
          gradedCount: Number(selectedTest.gradedCount || 0) + 1,
          lastResult: mergeLastResultWithRecords(selectedTest.lastResult, data, [singleRecord]),
        });
        clearPickedMedia();
        setDetailTab("stats");
        setSuccessMessage("Chấm xong. Đã chuyển sang tab Thống kê.");
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Chấm điểm thất bại.");
    } finally {
      setSubmitting(false);
    }
  }

  function renderStatsPanel(selectionMode: "inline" | "route") {
    return (
      <StatsPanel
        selectionMode={selectionMode}
        filteredGradeRecords={filteredGradeRecords}
        gradeRecords={gradeRecords}
        statsFilterMode={statsFilterMode}
        statsFilterValue={statsFilterValue}
        setStatsFilterMode={setStatsFilterMode}
        setStatsFilterValue={setStatsFilterValue}
        selectedGradeRecord={selectedGradeRecord}
        handwritingOverlayRois={handwritingOverlayRois}
        onOpenRecordDetail={openRecordDetail}
        onDeleteGradeRecord={deleteGradeRecord}
      />
    );
  }

  const buildSampleUrl = (sampleFile: string) => `${API_CONFIG.BASE_URL}/static/omr_data/${encodeURIComponent(sampleFile)}`;

  return (
    <div className="omr-mobile-shell">
      <header className="omr-header">
        <div className="omr-header-side">
          {isRecordDetailPage ? (
            <button
              className="header-back-btn tap-feedback"
              onClick={() => {
                setSelectedGradeRecordId(null);
                setResultImageUrl(null);
                setViewImage(null);
                stopCamera();
                navigate("/multichoice");
              }}
            >
              ← Danh sách
            </button>
          ) : showDetail ? (
            <button
              className="header-back-btn tap-feedback"
              onClick={() => {
                setDetailTestId(null);
                setNavTab("home");
                setResultImageUrl(null);
                setViewImage(null);
                clearPickedMedia();
                stopCamera();
                navigate("/multichoice");
              }}
            >
              ← Trang chủ
            </button>
          ) : navTab !== "home" ? (
            <button className="header-back-btn tap-feedback" onClick={() => setNavTab("home")}>
              ← Trang chủ
            </button>
          ) : (
            <button
              className="header-back-btn tap-feedback"
              onClick={() => {
                stopCamera();
                navigate("/home");
              }}
            >
              ← HomePage
            </button>
          )}
        </div>
        <div className="omr-header-title">
          {isRecordDetailPage
            ? "Thông tin chi tiết bản ghi"
            : showDetail
              ? "Thông tin bài thi"
              : navTab === "home"
                ? "Trang chủ OMR"
                : "Kho mẫu OMR"}
        </div>
        <div className="omr-header-side" style={{ justifyContent: "flex-end" }} />
      </header>

      <main className="omr-main">
        {isRecordDetailPage && (
          <section className="omr-section">
            {!selectedTest ? (
              <div className="empty-box">Không tìm thấy bài thi hoặc chưa có dữ liệu bản ghi.</div>
            ) : (
              <>
                <div className="detail-meta">Tên bài: {selectedTest.title}</div>
                <div className="detail-meta">Ngày tạo: {selectedTest.createdAt || "-"}</div>
                <div className="detail-meta">Tổng số mã đề: {selectedTest.answerSets.length}</div>
                {renderStatsPanel("route")}
              </>
            )}
          </section>
        )}

        {!isRecordDetailPage && !showDetail && navTab === "home" && (
          <section className="omr-section">
            <h3 className="section-title">Tổng quan nhanh</h3>
            <div className="home-note">Hiện có {tests.length} bài thi đã tạo. Mỗi bài thi có thể chứa nhiều mã đề.</div>
            <div className="home-note">
              Bài kiểm tra có thể chứa nhiều mã đề. Hãy tạo bài kiểm tra trước, sau đó thêm mã đề trong tab Đáp án.
            </div>
            <div className="home-list-wrap">
              <h4 className="home-list-title">Danh sách bài thi</h4>
              {tests.length === 0 && <div className="empty-box">Chưa có bài kiểm tra nào. Bấm + để tạo mới.</div>}
              {tests.map((item) => (
                <div
                  key={`home_${item.id}`}
                  className="test-card tap-feedback"
                  onClick={() => openTestForGrading(item.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      openTestForGrading(item.id);
                    }
                  }}
                >
                  <div className="test-card-head">
                    <div className="test-title">{item.title}</div>
                    <div className="template-tag">{item.answerSets.length} mã đề</div>
                  </div>
                  <div className="test-sub">Ngày tạo: {item.createdAt || "-"}</div>
                  <div className="test-sub">Tổng số mã đề: {item.answerSets.length}</div>
                  <div className="test-card-actions">
                    <button
                      type="button"
                      className="mini-btn danger tap-feedback test-card-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteAssignment(item.id);
                      }}
                    >
                      Xóa bài thi
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {!isRecordDetailPage && !showDetail && navTab === "templates" && (
          <section className="omr-section">
            <h4 className="home-list-title">Phiếu mẫu sẵn có</h4>
            {formProfilesLoading && <div className="empty-box">Đang tải profile phiếu mẫu...</div>}
            {!formProfilesLoading && formProfiles.length === 0 && <div className="empty-box">Chưa có profile phiếu mẫu trong omr_data.</div>}
            {!formProfilesLoading && formProfiles.map((p) => (
              <div
                key={`profile_${p.code}`}
                className="template-card tap-feedback"
                role="button"
                tabIndex={0}
                onClick={() => setSamplePreview({ title: p.title, fileName: p.sample_file, url: buildSampleUrl(p.sample_file) })}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSamplePreview({ title: p.title, fileName: p.sample_file, url: buildSampleUrl(p.sample_file) });
                  }
                }}
              >
                <div>
                  <div className="test-title">{p.title}</div>
                  <div className="test-sub">Mã đề: 3 số</div>
                  <div className="test-sub">MSSV: {Math.max(1, Number(p.student_id_digits || 6))} số</div>
                  <div className="test-sub">Câu mặc định: {p.default_questions} | Điểm mặc định: 10</div>
                </div>
                <div className="template-actions">
                  <button
                    type="button"
                    className="mini-btn tap-feedback"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate("/admin");
                    }}
                  >
                    → Admin
                  </button>
                  <button
                    type="button"
                    className="mini-btn tap-feedback"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSamplePreview({ title: p.title, fileName: p.sample_file, url: buildSampleUrl(p.sample_file) });
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
        )}

        {showDetail && selectedTest && (
          <section className="omr-section">
            <div className="detail-meta">Tạo lúc: {selectedTest.createdAt}</div>
            <div className="detail-meta">Profile: {selectedTest.formProfileCode || "(mặc định)"}</div>

            <div className="top-tab-scroll">
              <button className={`top-tab tap-feedback ${detailTab === "grading" ? "active" : ""}`} onClick={() => setDetailTab("grading")}>Chấm bài</button>
              <button className={`top-tab tap-feedback ${detailTab === "answers" ? "active" : ""}`} onClick={() => setDetailTab("answers")}>Đáp án</button>
              <button className={`top-tab tap-feedback ${detailTab === "stats" ? "active" : ""}`} onClick={() => setDetailTab("stats")}>Thống kê</button>
              <button className={`top-tab tap-feedback ${detailTab === "export" ? "active" : ""}`} onClick={() => setDetailTab("export")}>Xuất file</button>
            </div>

            {detailTab === "grading" && (
              <div>
                <div className="smart-scanner-box">
                  <div
                    className="smart-camera-stage"
                    ref={cameraStageRef}
                    style={{ aspectRatio: `1 / ${scannerAspectRatio}` }}
                  >
                    <video
                      ref={videoRef}
                      className={`camera-video ${cameraOn ? "" : "hidden"}`}
                      playsInline
                      muted
                      autoPlay
                    />
                    {!cameraOn && <div className="smart-camera-placeholder">Đang chờ camera...</div>}

                    <div className="smart-overlay">
                      <div
                        ref={viewfinderRef}
                        className={`smart-viewfinder portrait ${scannerState === "locked" ? "match" : ""}`}
                        style={{ aspectRatio: `1 / ${scannerAspectRatio}` }}
                      >
                        <span className="bracket tl" />
                        <span className="bracket tr" />
                        <span className="bracket bl" />
                        <span className="bracket br" />
                        {CORNER_KEYS.map((corner) => {
                          const marker = markerSamples[corner];
                          return (
                            <span
                              key={`target-${corner}`}
                              className={`target-box ${corner}`}
                              style={{
                                left: `${(marker.x * 100).toFixed(2)}%`,
                                top: `${(marker.y * 100).toFixed(2)}%`,
                              }}
                            />
                          );
                        })}

                        {scannerState === "searching" && (
                          <div className="smart-lock-text">Đang dò 4 ô đen theo mẫu PDF...</div>
                        )}
                        {scannerState === "locked" && (
                          <div className="smart-lock-text blink">Đã nhận đúng 4 marker, đang tự chụp...</div>
                        )}
                      </div>
                    </div>

                  </div>
                </div>

                <div className="scanner-actions smart-controls">
                  <button className="library-btn tap-feedback scanner-library-btn" onClick={() => libraryInputRef.current?.click()}>
                    Tải ảnh từ thư viện
                  </button>
                  <button className="ghost-btn tap-feedback" onClick={() => void startSmartScannerFlow()}>
                    {cameraOn ? "Làm mới camera" : "Bật camera"}
                  </button>
                  {cameraOn && (
                    <button className="ghost-btn tap-feedback" onClick={stopCamera}>
                      Tắt camera
                    </button>
                  )}
                </div>

                {cameraOn && scannerState !== "locked" && (
                  <div className="scanner-hint">Cần căn đủ 4 góc đen vào khung để hệ thống tự chụp.</div>
                )}

                {cameraOn && scannerState === "locked" && (
                  <div className="scanner-hint">Khung đã xanh, hệ thống đang tự chụp ảnh.</div>
                )}

                {cameraError && <div className="error-toast">{cameraError}</div>}

                {selectedTestProfile?.strategy?.sheet_aspect_ratio && (
                  <div className="scanner-profile-note">
                    Khung nhận diện theo mẫu PDF ({selectedTestProfile.sample_file}): tỷ lệ {selectedTestProfile.strategy.sheet_aspect_ratio.toFixed(3)}.
                  </div>
                )}

                <input
                  ref={libraryInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => onPickFiles(e.target.files)}
                  style={{ display: "none" }}
                />

                {pickedPreviews.length > 0 && (
                  <div className="omr-preview-strip">
                    <div className="omr-preview-row">
                      {pickedPreviews.map((url, idx) => (
                        <div key={`${url}-${idx}`} className="omr-preview-item">
                          <span className={`preview-source-badge ${pickedSource || "library"}`}>
                            {pickedSource === "camera" ? "Camera" : pickedSource === "mixed" ? "Hỗn hợp" : "Thư viện"}
                          </span>
                          <img src={url} alt={`scan-${idx}`} onClick={() => setViewImage(url)} />
                          <button className="omr-remove-preview" onClick={() => removePickedFile(idx)}>×</button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <button className="primary-btn tap-feedback" onClick={gradeCurrentTest} disabled={!canGradeNow}>
                  {submitting ? "Đang chấm..." : "Chấm bài ngay"}
                </button>

                {!canGradeNow && !submitting && (
                  <div className="grade-disabled-hint">{gradeBlockReason}</div>
                )}

                {resultImageUrl && (
                  <button className="ghost-btn tap-feedback" onClick={() => setViewImage(resultImageUrl)}>
                    Xem ảnh kết quả chấm
                  </button>
                )}
              </div>
            )}

            {detailTab === "answers" && (
              <div className="answer-list">
                <div className="code-toolbar">
                  <input
                    className="code-input"
                    value={newCode}
                    onChange={(e) => setNewCode(e.target.value.replace(/\D/g, "").slice(0, 3))}
                    placeholder="Nhập mã đề 3 số (VD: 001)"
                    inputMode="numeric"
                    pattern="\\d{3}"
                    maxLength={3}
                  />
                  <button className="code-add-btn tap-feedback" onClick={addAnswerCode}>Thêm mã đề</button>
                </div>

                {selectedTest.answerSets.length > 0 && (
                  <div className="answer-code-section">
                    <div className="answer-code-title">Danh sách Mã đề</div>
                    <div className="variant-switch-row">
                      {selectedTest.answerSets.map((set) => (
                        <button
                          key={set.code}
                          className={`variant-chip tap-feedback ${selectedTest.activeCode === set.code ? "active" : ""}`}
                          onClick={() => updateSelectedTest({ activeCode: set.code })}
                        >
                          Mã đề {set.code}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {selectedTest.answerSets.length === 0 && (
                  <div className="empty-box">Chưa có mã đề. Hãy thêm mã đề trước khi nhập đáp án.</div>
                )}

                {selectedAnswerSet && Array.from({ length: selectedTest.questionCount }).map((_, idx) => (
                  <div key={idx} className="answer-row">
                    <div className="answer-q">Câu {idx + 1}</div>
                    <div className="answer-options">
                      {OPTION_LABELS.map((opt) => (
                        <button
                          key={opt}
                          className={`option-btn tap-feedback ${selectedAnswerSet.answers[idx] === opt ? "active" : ""}`}
                          onClick={() => updateAnswer(idx, opt)}
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}

                {selectedTest.answerSets.length > 0 && (
                  <div className="variant-remove-row">
                    {selectedTest.answerSets.map((set) => (
                      <button key={`remove_${set.code}`} className="variant-chip tap-feedback" onClick={() => removeAnswerCode(set.code)}>
                        Xóa mã {set.code}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {detailTab === "stats" && (
              <>
                {batchSummary && (
                  <div className="batch-summary-banner">
                    <span className="batch-summary-text">
                      Batch: <strong>{batchSummary.successCount}/{batchSummary.totalCount}</strong> ảnh thành công
                      {batchSummary.failedCount > 0 && (
                        <span className="batch-failed-count"> · {batchSummary.failedCount} ảnh lỗi</span>
                      )}
                    </span>
                    {batchSummary.zipUrl && (
                      <a
                        className="mini-btn template-download-btn batch-zip-btn"
                        href={batchSummary.zipUrl}
                        download
                        target="_blank"
                        rel="noreferrer"
                      >
                        ⬇ Tải ZIP overlay
                      </a>
                    )}
                  </div>
                )}
                {renderStatsPanel("inline")}
              </>
            )}

            {detailTab === "export" && (
              <ExportPanel
                selectedTest={selectedTest}
                gradeRecords={gradeRecords}
                onError={setErrorMessage}
                onSuccess={setSuccessMessage}
              />
            )}
          </section>
        )}
      </main>

      {!isRecordDetailPage && !showDetail && navTab === "home" && (
        <button className="fab tap-feedback" onClick={() => setSheetOpen(true)}>
          +
        </button>
      )}

      {!isRecordDetailPage && <nav className="bottom-nav">
        <button
          className={`nav-btn tap-feedback ${navTab === "home" ? "active" : ""}`}
          onClick={() => {
            setDetailTestId(null);
            setNavTab("home");
            setResultImageUrl(null);
            setViewImage(null);
            clearPickedMedia();
            stopCamera();
            navigate("/multichoice");
          }}
        >
          Trang chủ
        </button>
        <button
          className={`nav-btn tap-feedback ${navTab === "templates" ? "active" : ""}`}
          onClick={() => {
            setDetailTestId(null);
            setNavTab("templates");
            setResultImageUrl(null);
            setViewImage(null);
            clearPickedMedia();
            stopCamera();
            navigate("/multichoice");
          }}
        >
          Mẫu có sẵn
        </button>
      </nav>}

      {sheetOpen && (
        <>
          <div className="sheet-overlay show" onClick={() => setSheetOpen(false)} />
          <section className="bottom-sheet show" role="dialog" aria-modal="true" aria-label="Tạo bài kiểm tra mới">
            <div className="sheet-handle" />
            <h3 className="sheet-title">Tạo bài kiểm tra mới</h3>
            <div className="sheet-steps">
              <span className="step-chip active">1. Thông tin</span>
              <span className="step-chip">2. Cấu hình</span>
              <span className="step-chip">3. Hoàn tất</span>
            </div>

            <label className="field-label">Tên bài kiểm tra</label>
            <input
              className="field-input"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              placeholder="VD: Kiểm tra chương 3 - Toán 9A"
            />

            <label className="field-label">Thời gian tạo bài kiểm tra</label>
            <input
              className="field-input"
              type="datetime-local"
              value={newCreatedAt}
              onChange={(e) => setNewCreatedAt(e.target.value)}
            />

            <label className="field-label">Profile phiếu mẫu (do dev cấu hình)</label>
            <select
              className="field-input"
              value={newFormProfileCode}
              onChange={(e) => setNewFormProfileCode(e.target.value)}
            >
              {formProfiles.map((p) => (
                <option key={p.code} value={p.code}>
                  {p.title} ({p.code})
                </option>
              ))}
            </select>

            <div className="accordion open">
              <div className="stepper-row">
                <span>Số câu mặc định</span>
                <strong>{selectedNewProfile?.default_questions || "-"}</strong>
              </div>

              <div className="stepper-row">
                <span>Điểm tổng mặc định</span>
                <strong>10</strong>
              </div>
            </div>

            <button className="primary-btn tap-feedback" onClick={createTestFromSheet}>Tạo bài kiểm tra</button>
          </section>
        </>
      )}

      {samplePreview && (
        <div className="sample-preview-overlay" onClick={() => setSamplePreview(null)}>
          <div className="sample-preview-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="sample-preview-head">
              <strong>{samplePreview.title}</strong>
              <button className="header-back-btn tap-feedback" onClick={() => setSamplePreview(null)}>Đóng</button>
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

      {!!errorMessage && <div className="error-toast">{errorMessage}</div>}
      {!!successMessage && <div className="success-toast">{successMessage}</div>}
      {viewImage && <ViewImageModal img={viewImage} onClose={() => setViewImage(null)} />}
    </div>
  );
}
