import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_CONFIG } from "../config/api";
import ViewImageModal from "../UI_Components/ViewImageModal";
import "../UI_Components/OmrMobileApp.css";

type NavTab = "home" | "tests" | "templates";
type DetailTab = "grading" | "answers" | "stats" | "export";
type ScannerUiState = "idle" | "searching" | "locked";
type PickedImageSource = "camera" | "library" | "mixed" | null;

type FormProfile = {
  code: string;
  title: string;
  sample_file: string;
  default_questions: number;
  total_points: number;
  num_choices: number;
  rows_per_block: number;
  num_blocks?: number | null;
  exam_code_digits?: number;
  student_id_digits: number;
  sid_has_write_row: boolean;
  strategy?: ScannerStrategy;
};

type CornerKey = "tl" | "tr" | "bl" | "br";

type MarkerBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  cx?: number;
  cy?: number;
};

type ScannerHint = {
  min_dark_ratio?: number;
  min_center_luma?: number;
  min_marker_contrast?: number;
  sample_size_norm?: number;
};

type ScannerStrategy = {
  sheet_aspect_ratio?: number;
  corner_markers?: Partial<Record<CornerKey, MarkerBox>>;
  scanner_hint?: ScannerHint;
};

type MarkerSample = {
  x: number;
  y: number;
  sampleSize: number;
};

type AnswerSet = {
  code: string;
  answers: string[];
};

type AnswerCompareItem = {
  question?: number;
  selected_label?: string;
  correct_label?: string;
  status?: string;
  is_correct?: boolean;
};

type OMRResult = {
  score?: number;
  graded_questions?: number;
  uncertain_count?: number;
  answer_compare?: AnswerCompareItem[];
  student_id?: string;
  exam_code?: string;
  image_url?: string | null;
  sid_crop_url?: string | null;
  mcq_crop_url?: string | null;
  bubble_confidence_json_url?: string | null;
  result_image?: string;
  sid_crop_image?: string;
  mcq_crop_image?: string;
  bubble_confidence_json?: string;
  [key: string]: unknown;
};

type BatchGradeResultItem = {
  file_name?: string;
  success?: boolean;
  data?: OMRResult;
  image_url?: string | null;
  sid_crop_url?: string | null;
  mcq_crop_url?: string | null;
  bubble_confidence_json_url?: string | null;
};

type GradeRecord = {
  id: string;
  graded_at: string;
  source: "single" | "batch";
  file_name: string;
  image_url?: string | null;
  sid_crop_url?: string | null;
  mcq_crop_url?: string | null;
  bubble_confidence_json_url?: string | null;
  data: OMRResult;
};

type LastResultPayload = OMRResult & {
  grade_records?: GradeRecord[];
  __meta__?: {
    form_profile_code?: string;
    [key: string]: unknown;
  };
};

type TestCardItem = {
  id: number;
  title: string;
  createdAt: string;
  createdAtRaw?: string;
  questionCount: number;
  totalPoints: number;
  formProfileCode: string | null;
  gradedCount: number;
  answerSets: AnswerSet[];
  activeCode: string | null;
  lastResult?: LastResultPayload | null;
};

type AssignmentApiItem = {
  aid: number;
  title: string;
  created_at_raw?: string | null;
  created_at_label?: string | null;
  question_count: number;
  total_points: number;
  form_profile_code?: string | null;
  graded_count: number;
  answer_sets: AnswerSet[];
  active_code?: string | null;
  last_result?: LastResultPayload | null;
};

const OPTION_LABELS = ["A", "B", "C", "D"];
const CORNER_KEYS: CornerKey[] = ["tl", "tr", "bl", "br"];
const MAX_GRADE_RECORDS = 200;
const AUTO_CAPTURE_COOLDOWN_MS = 1300;

const FALLBACK_MARKER_SAMPLES: Record<CornerKey, MarkerSample> = {
  tl: { x: 0.08, y: 0.08, sampleSize: 0.062 },
  tr: { x: 0.92, y: 0.08, sampleSize: 0.062 },
  bl: { x: 0.08, y: 0.92, sampleSize: 0.062 },
  br: { x: 0.92, y: 0.92, sampleSize: 0.062 },
};

const clampNumber = (value: number, min: number, max: number) => {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
};

function mapClientPointToVideo(
  clientX: number,
  clientY: number,
  videoEl: HTMLVideoElement,
  videoRect: DOMRect
) {
  const sourceW = videoEl.videoWidth;
  const sourceH = videoEl.videoHeight;
  const drawScale = Math.max(videoRect.width / sourceW, videoRect.height / sourceH);
  const drawW = sourceW * drawScale;
  const drawH = sourceH * drawScale;
  const offsetX = (videoRect.width - drawW) / 2;
  const offsetY = (videoRect.height - drawH) / 2;

  const xInVideo = (clientX - videoRect.left - offsetX) / drawScale;
  const yInVideo = (clientY - videoRect.top - offsetY) / drawScale;

  return {
    x: Math.max(0, Math.min(sourceW - 1, xInVideo)),
    y: Math.max(0, Math.min(sourceH - 1, yInVideo)),
  };
}

function darkRatioInArea(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  darkLumaThreshold: number
) {
  const sx = Math.max(0, Math.floor(x - size / 2));
  const sy = Math.max(0, Math.floor(y - size / 2));
  const sw = Math.max(1, Math.floor(size));
  const sh = Math.max(1, Math.floor(size));
  const { data } = ctx.getImageData(sx, sy, sw, sh);
  let dark = 0;
  let total = 0;
  for (let i = 0; i < data.length; i += 4) {
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    if (lum < darkLumaThreshold) dark += 1;
    total += 1;
  }
  return total ? dark / total : 0;
}

function avgLumaInArea(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number
) {
  const sx = Math.max(0, Math.floor(x));
  const sy = Math.max(0, Math.floor(y));
  const sw = Math.max(1, Math.floor(w));
  const sh = Math.max(1, Math.floor(h));
  const { data } = ctx.getImageData(sx, sy, sw, sh);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    count += 1;
  }
  return count ? sum / count : 0;
}

const nowDateTimeLocal = () => {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
};

const formatDateTimeLabel = (value: string) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
};

const toAnswerString = (answers: string[]) => {
  const oneBased = answers.map((x) => {
    const idx = OPTION_LABELS.indexOf(x);
    return idx >= 0 ? String(idx + 1) : "";
  });
  return oneBased.join(",");
};

const toAbsoluteStaticUrl = (url: string | null | undefined) => {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_CONFIG.BASE_URL}${url}`;
};

const toStaticOmrUrlFromFileName = (fileName: unknown) => {
  if (typeof fileName !== "string") return null;
  const safeName = fileName.trim();
  if (!safeName) return null;
  return toAbsoluteStaticUrl(`/static/omr/${encodeURIComponent(safeName)}`);
};

const createRecordId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
};

const normalizeRecordUrls = (record: GradeRecord): GradeRecord => {
  const dataPayload = (record.data && typeof record.data === "object") ? record.data : {};
  const fallbackImage = toStaticOmrUrlFromFileName((dataPayload as OMRResult).result_image);
  const fallbackSid = toStaticOmrUrlFromFileName((dataPayload as OMRResult).sid_crop_image);
  const fallbackMcq = toStaticOmrUrlFromFileName((dataPayload as OMRResult).mcq_crop_image);
  const fallbackTelemetry = toStaticOmrUrlFromFileName((dataPayload as OMRResult).bubble_confidence_json);

  return {
    ...record,
    image_url: toAbsoluteStaticUrl(record.image_url) || fallbackImage,
    sid_crop_url: toAbsoluteStaticUrl(record.sid_crop_url) || fallbackSid,
    mcq_crop_url: toAbsoluteStaticUrl(record.mcq_crop_url) || fallbackMcq,
    bubble_confidence_json_url:
      toAbsoluteStaticUrl(record.bubble_confidence_json_url)
      || toAbsoluteStaticUrl((dataPayload as OMRResult).bubble_confidence_json_url)
      || fallbackTelemetry,
  };
};

const buildLegacyRecordFromLastResult = (
  lastResult: LastResultPayload | OMRResult | null | undefined
): GradeRecord | null => {
  if (!lastResult || typeof lastResult !== "object") return null;

  const payload = lastResult as OMRResult;
  const imageUrl = toAbsoluteStaticUrl(payload.image_url) || toStaticOmrUrlFromFileName(payload.result_image);
  const sidCropUrl = toAbsoluteStaticUrl(payload.sid_crop_url) || toStaticOmrUrlFromFileName(payload.sid_crop_image);
  const mcqCropUrl = toAbsoluteStaticUrl(payload.mcq_crop_url) || toStaticOmrUrlFromFileName(payload.mcq_crop_image);
  const bubbleConfidenceJsonUrl =
    toAbsoluteStaticUrl(payload.bubble_confidence_json_url)
    || toStaticOmrUrlFromFileName(payload.bubble_confidence_json);

  if (!imageUrl && !sidCropUrl && !mcqCropUrl && !bubbleConfidenceJsonUrl) {
    return null;
  }

  const seed = String(
    payload.result_image
    || payload.sid_crop_image
    || payload.mcq_crop_image
    || payload.exam_code
    || payload.student_id
    || "legacy"
  );

  return {
    id: `legacy_${seed}`,
    graded_at: "1970-01-01T00:00:00.000Z",
    source: "single",
    file_name: String(payload.result_image || payload.sid_crop_image || payload.mcq_crop_image || "legacy_result"),
    image_url: imageUrl,
    sid_crop_url: sidCropUrl,
    mcq_crop_url: mcqCropUrl,
    bubble_confidence_json_url: bubbleConfidenceJsonUrl,
    data: payload,
  };
};

const getGradeRecordsFromLastResult = (lastResult: LastResultPayload | OMRResult | null | undefined): GradeRecord[] => {
  const records = (lastResult as LastResultPayload | null | undefined)?.grade_records;
  if (Array.isArray(records)) {
    const normalized = records.filter((item): item is GradeRecord => {
      return (
        !!item
        && typeof item === "object"
        && typeof item.id === "string"
        && typeof item.graded_at === "string"
        && typeof item.file_name === "string"
        && !!item.data
        && typeof item.data === "object"
      );
    }).map(normalizeRecordUrls);

    if (normalized.length > 0) {
      return normalized;
    }
  }

  const legacyRecord = buildLegacyRecordFromLastResult(lastResult);
  return legacyRecord ? [legacyRecord] : [];
};

const buildGradeRecord = ({
  source,
  fileName,
  imageUrl,
  sidCropUrl,
  mcqCropUrl,
  bubbleConfidenceJsonUrl,
  data,
}: {
  source: "single" | "batch";
  fileName: string;
  imageUrl?: string | null;
  sidCropUrl?: string | null;
  mcqCropUrl?: string | null;
  bubbleConfidenceJsonUrl?: string | null;
  data: OMRResult;
}): GradeRecord => {
  return {
    id: createRecordId(),
    graded_at: new Date().toISOString(),
    source,
    file_name: String(fileName || "omr_submission.jpg"),
    image_url: imageUrl ?? null,
    sid_crop_url: sidCropUrl ?? null,
    mcq_crop_url: mcqCropUrl ?? null,
    bubble_confidence_json_url: bubbleConfidenceJsonUrl ?? null,
    data,
  };
};

const mergeLastResultWithRecords = (
  previous: LastResultPayload | OMRResult | null | undefined,
  latestData: OMRResult,
  newRecords: GradeRecord[]
): LastResultPayload => {
  const base = (previous && typeof previous === "object") ? (previous as LastResultPayload) : {};
  const previousRecords = getGradeRecordsFromLastResult(previous);
  return {
    ...base,
    ...latestData,
    grade_records: [...newRecords, ...previousRecords].slice(0, MAX_GRADE_RECORDS),
  };
};

const rebuildLastResultFromRecords = (
  previous: LastResultPayload | OMRResult | null | undefined,
  records: GradeRecord[]
): LastResultPayload => {
  const base = (previous && typeof previous === "object") ? (previous as LastResultPayload) : {};
  const safeRecords = records.slice(0, MAX_GRADE_RECORDS);

  if (safeRecords.length <= 0) {
    return {
      __meta__: base.__meta__,
      grade_records: [],
    };
  }

  return {
    ...base,
    ...safeRecords[0].data,
    grade_records: safeRecords,
  };
};

const getCompareStatus = (item: AnswerCompareItem | null | undefined) => String(item?.status || "").trim().toLowerCase();

const isScoredCompareStatus = (status: string) => status !== "blank-no-key" && status !== "no-key";

const countUncertainWithKey = (result: OMRResult | null | undefined) => {
  const compare = Array.isArray(result?.answer_compare) ? result.answer_compare : [];
  const fromCompare = compare.filter((x) => getCompareStatus(x) === "uncertain").length;
  if (fromCompare > 0 || compare.length > 0) {
    return fromCompare;
  }
  return Number(result?.uncertain_count || 0);
};

function toCsv(result: OMRResult | null | undefined): string {
  const rows = [["Cau", "Da_chon", "Dung", "Trang_thai", "Dung/Sai"]];
  const compare = Array.isArray(result?.answer_compare) ? result.answer_compare : [];
  compare.forEach((it: AnswerCompareItem) => {
    rows.push([
      String(it.question ?? ""),
      String(it.selected_label ?? ""),
      String(it.correct_label ?? ""),
      String(it.status ?? ""),
      it.is_correct ? "1" : "0",
    ]);
  });
  return rows.map((r) => r.map((c) => `"${String(c).replaceAll('"', '""')}"`).join(",")).join("\n");
}

function downloadTextFile(fileName: string, content: string, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function MultichoicePage() {
  const navigate = useNavigate();
  const uid = Number(localStorage.getItem("uid") || "0");

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

  const [pickedFiles, setPickedFiles] = useState<File[]>([]);
  const [pickedPreviews, setPickedPreviews] = useState<string[]>([]);
  const [pickedSource, setPickedSource] = useState<PickedImageSource>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
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

  const selectedTest = useMemo(
    () => tests.find((x) => x.id === detailTestId) || null,
    [tests, detailTestId]
  );

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

  const showDetail = !!selectedTest;

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

  const selectedGradeRecord = useMemo(() => {
    if (!gradeRecords.length) return null;
    if (!selectedGradeRecordId) return gradeRecords[0];
    return gradeRecords.find((record) => record.id === selectedGradeRecordId) || gradeRecords[0];
  }, [gradeRecords, selectedGradeRecordId]);

  const statsResult = selectedGradeRecord?.data || selectedTest?.lastResult || null;
  const selectedTelemetryJsonUrl = useMemo(() => {
    if (!selectedGradeRecord) return null;
    return (
      toAbsoluteStaticUrl(selectedGradeRecord.bubble_confidence_json_url)
      || toAbsoluteStaticUrl(selectedGradeRecord.data?.bubble_confidence_json_url)
      || toStaticOmrUrlFromFileName(selectedGradeRecord.data?.bubble_confidence_json)
    );
  }, [selectedGradeRecord]);

  const statData = useMemo(() => {
    const compare = Array.isArray(statsResult?.answer_compare)
      ? statsResult?.answer_compare
      : [];
    const scoredCompare = compare.filter((x: AnswerCompareItem) => {
      const status = getCompareStatus(x);
      return isScoredCompareStatus(status);
    });

    const total = Math.max(0, Number(statsResult?.graded_questions ?? scoredCompare.length));
    const correct = scoredCompare.filter((x: AnswerCompareItem) => getCompareStatus(x) === "correct" || !!x.is_correct).length;
    const uncertain = countUncertainWithKey(statsResult || null);
    const wrong = Math.max(0, total - correct - uncertain);
    return { total, correct, wrong, uncertain };
  }, [statsResult]);

  const pieStyle = useMemo(() => {
    const total = Math.max(1, statData.total);
    const c = Math.round((statData.correct / total) * 100);
    const u = Math.round((statData.uncertain / total) * 100);
    const w = Math.max(0, 100 - c - u);
    return {
      background: `conic-gradient(#2e7d32 0 ${c}%, #f9a825 ${c}% ${c + u}%, #d32f2f ${c + u}% ${c + u + w}%)`,
    } as const;
  }, [statData]);

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
    if (gradeRecords.length === 0) {
      if (selectedGradeRecordId) {
        setSelectedGradeRecordId(null);
      }
      return;
    }
    if (!selectedGradeRecordId || !gradeRecords.some((record) => record.id === selectedGradeRecordId)) {
      setSelectedGradeRecordId(gradeRecords[0].id);
    }
  }, [gradeRecords, selectedGradeRecordId]);

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
      setNavTab("tests");
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
      answers[questionIdx] = value;
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
        setDetailTab("stats");
        setSuccessMessage("Chấm xong. Đã chuyển sang tab Thống kê.");
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Chấm điểm thất bại.");
    } finally {
      setSubmitting(false);
    }
  }

  function exportExcel() {
    if (!selectedTest) {
      setErrorMessage("Chưa chọn bài kiểm tra để xuất file.");
      return;
    }
    const exportResult = selectedGradeRecord?.data || selectedTest.lastResult;
    if (!exportResult) {
      setErrorMessage("Chưa có dữ liệu để xuất Excel.");
      return;
    }
    const csv = toCsv(exportResult);
    downloadTextFile(`${selectedTest.title.replace(/\s+/g, "_")}_ket_qua.csv`, csv, "text/csv;charset=utf-8");
  }

  function exportPdfSummary() {
    if (!selectedTest) return;
    const r = selectedGradeRecord?.data || selectedTest.lastResult;
    const uncertainWithKey = countUncertainWithKey(r || null);
    const codeLabel = selectedAnswerSet ? `Mã đề ${selectedAnswerSet.code}` : "-";
    const lines = [
      `Bao cao OMR - ${selectedTest.title}`,
      `Ngay tao: ${selectedTest.createdAt}`,
      `Bo dap an: ${codeLabel}`,
      `Anh bai lam: ${selectedGradeRecord?.file_name || "-"}`,
      `So cau: ${selectedTest.questionCount}`,
      `Tong diem: ${r?.score ?? "-"}`,
      `So cau da cham: ${r?.graded_questions ?? "-"}`,
      `Cau uncertain: ${uncertainWithKey}`,
      "",
      "Ghi chu: Bao cao nhanh dang van ban cho mobile UI.",
    ];
    downloadTextFile(`${selectedTest.title.replace(/\s+/g, "_")}_bao_cao.txt`, lines.join("\n"));
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
        setNavTab("tests");
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Xóa bài thi thất bại");
    }
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

  function clearGradeHistory() {
    if (!selectedTest) return;
    if (gradeRecords.length <= 0) return;

    if (!window.confirm(`Xóa toàn bộ ${gradeRecords.length} bản ghi đã chấm của bài thi này?`)) return;

    const nextLastResult = rebuildLastResultFromRecords(selectedTest.lastResult, []);
    setSelectedGradeRecordId(null);
    setResultImageUrl(null);
    updateSelectedTest({
      gradedCount: 0,
      lastResult: nextLastResult,
    });
    setSuccessMessage("Đã xóa toàn bộ lịch sử chấm.");
  }

  const buildSampleUrl = (sampleFile: string) => `${API_CONFIG.BASE_URL}/static/omr_data/${encodeURIComponent(sampleFile)}`;

  return (
    <div className="omr-mobile-shell">
      <header className="omr-header">
        <div className="omr-header-side">
          {showDetail ? (
            <button
              className="header-back-btn tap-feedback"
              onClick={() => {
                setDetailTestId(null);
                setNavTab("tests");
                stopCamera();
              }}
            >
              ← DS bài thi
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
          {showDetail ? `Chấm điểm: ${selectedTest.title}` : navTab === "home" ? "Trang chủ OMR" : navTab === "tests" ? "Danh sách bài thi" : "Kho mẫu OMR"}
        </div>
        <div className="omr-header-side" style={{ justifyContent: "flex-end" }}>
          {showDetail ? (
            <button
              className="header-back-btn tap-feedback"
              onClick={() => {
                setDetailTestId(null);
                setNavTab("home");
                stopCamera();
              }}
            >
              Trang chủ
            </button>
          ) : navTab === "home" ? (
            <button className="header-back-btn tap-feedback" onClick={() => setNavTab("tests")}>
              Bài thi
            </button>
          ) : null}
        </div>
      </header>

      <main className="omr-main">
        {!showDetail && navTab === "home" && (
          <section className="omr-section">
            <h3 className="section-title">Tổng quan nhanh</h3>
            <div className="home-note">Hiện có {tests.length} bài thi đã tạo. Mỗi bài thi có thể chứa nhiều mã đề.</div>
            <div className="home-note">
              Bài kiểm tra có thể chứa nhiều mã đề. Hãy tạo bài kiểm tra trước, sau đó thêm mã đề trong tab Đáp án.
            </div>
            <div className="home-quick-actions">
              <button
                className="mini-btn tap-feedback"
                onClick={() => {
                  stopCamera();
                  navigate("/home");
                }}
              >
                ← Về HomePage
              </button>
              <button className="mini-btn tap-feedback" onClick={() => { setNavTab("tests"); setSheetOpen(true); }}>
                + Tạo bài kiểm tra mới
              </button>
              <button className="mini-btn tap-feedback" onClick={() => setNavTab("tests")}>
                Xem danh sách bài kiểm tra
              </button>
            </div>
            <div className="home-list-wrap">
              <h4 className="home-list-title">Danh sách bài kiểm tra gần đây</h4>
              {tests.length === 0 && <div className="empty-box">Chưa có bài kiểm tra nào. Bấm + để tạo mới.</div>}
              {tests.slice(0, 4).map((item) => (
                <button
                  key={`home_${item.id}`}
                  className="test-card tap-feedback"
                  onClick={() => {
                    setNavTab("tests");
                    setDetailTestId(item.id);
                    setDetailTab("grading");
                  }}
                >
                  <div className="test-card-head">
                    <div className="test-title">{item.title}</div>
                    <div className="template-tag">{item.answerSets.length} mã đề</div>
                  </div>
                  <div className="test-sub">Tạo ngày: {item.createdAt}</div>
                  <div className="test-sub">Mã đề: {item.answerSets.map((x) => x.code).join(", ") || "(chưa thêm)"}</div>
                </button>
              ))}
            </div>
          </section>
        )}

        {!showDetail && navTab === "tests" && (
          <section className="omr-section">
            {tests.length === 0 && <div className="empty-box">Chưa có bài kiểm tra nào. Bấm nút + để tạo mới.</div>}
            {tests.map((item) => (
              <div key={item.id} className="test-card tap-feedback" onClick={() => {
                setDetailTestId(item.id);
                setDetailTab("grading");
                clearPickedMedia();
                setResultImageUrl(null);
                setErrorMessage("");
                stopCamera();
              }} role="button" tabIndex={0} onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setDetailTestId(item.id);
                  setDetailTab("grading");
                  clearPickedMedia();
                  setResultImageUrl(null);
                  setErrorMessage("");
                  stopCamera();
                }
              }}>
                <div className="test-card-head">
                  <div className="test-title">{item.title}</div>
                  <div className="template-tag">{item.answerSets.length} mã đề</div>
                </div>
                <div className="test-sub">Tạo ngày: {item.createdAt}</div>
                <div className="test-sub">Mã đề: {item.answerSets.map((x) => x.code).join(", ") || "(chưa thêm)"}</div>
                <div className="status-pill">Đã chấm {item.gradedCount}/{item.questionCount}</div>
                <div style={{ marginTop: "8px" }}>
                  <button
                    className="mini-btn danger tap-feedback"
                    onClick={(e) => {
                      e.stopPropagation();
                      void deleteAssignment(item.id);
                    }}
                    type="button"
                  >
                    Xóa bài thi
                  </button>
                </div>
              </div>
            ))}
          </section>
        )}

        {!showDetail && navTab === "templates" && (
          <section className="omr-section">
            <h4 className="home-list-title">Phiếu mẫu từ omr_data (dev cấu hình)</h4>
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
                  <div className="test-sub">Mã đề: {Math.max(1, Number(p.exam_code_digits || 3))} số</div>
                  <div className="test-sub">MSSV: {Math.max(1, Number(p.student_id_digits || 6))} số</div>
                  <div className="test-sub">Câu mặc định: {p.default_questions} | Điểm mặc định: 10</div>
                </div>
                <div className="template-actions">
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

            <div className="top-tab-scroll">
              <button className={`top-tab tap-feedback ${detailTab === "grading" ? "active" : ""}`} onClick={() => setDetailTab("grading")}>Chấm bài</button>
              <button className={`top-tab tap-feedback ${detailTab === "answers" ? "active" : ""}`} onClick={() => setDetailTab("answers")}>Đáp án</button>
              <button className={`top-tab tap-feedback ${detailTab === "stats" ? "active" : ""}`} onClick={() => setDetailTab("stats")}>Thống kê</button>
              <button className={`top-tab tap-feedback ${detailTab === "export" ? "active" : ""}`} onClick={() => setDetailTab("export")}>Xuất file</button>
            </div>

            {detailTab === "grading" && (
              <div>
                <div className="smart-scanner-box">
                  <div className="smart-camera-stage" ref={cameraStageRef}>
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
                    onChange={(e) => setNewCode(e.target.value.trim())}
                    placeholder="Nhập mã đề (VD: 001)"
                  />
                  <button className="code-add-btn tap-feedback" onClick={addAnswerCode}>Thêm mã đề</button>
                </div>

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
                  <div className="variant-switch-row">
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
              <div className="stats-wrap">
                <div className="stats-summary">
                  <div className="pie" style={pieStyle} />
                  <div className="legend">
                    <p><span className="dot ok" /> Đúng: {statData.correct}</p>
                    <p><span className="dot uncertain" /> Không chắc: {statData.uncertain}</p>
                    <p><span className="dot wrong" /> Sai: {statData.wrong}</p>
                    <p><span className="dot info" /> Điểm: {statsResult?.score ?? "-"}</p>
                  </div>
                </div>

                <div className="stats-records-layout">
                  <div className="stats-record-list">
                    <div className="stats-record-header">
                      <div className="stats-record-title">Danh sách bài đã chấm ({gradeRecords.length})</div>
                      <button
                        type="button"
                        className="mini-btn danger tap-feedback stats-record-clear-btn"
                        onClick={clearGradeHistory}
                        disabled={gradeRecords.length <= 0}
                      >
                        Xóa toàn bộ
                      </button>
                    </div>
                    {gradeRecords.length === 0 && (
                      <div className="empty-box">Chưa có bản ghi theo từng ảnh. Hãy chấm ít nhất 1 ảnh để tạo danh sách.</div>
                    )}
                    {gradeRecords.map((record) => (
                      <div
                        key={record.id}
                        className={`stats-record-item tap-feedback ${selectedGradeRecord?.id === record.id ? "active" : ""}`}
                        onClick={() => setSelectedGradeRecordId(record.id)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            setSelectedGradeRecordId(record.id);
                          }
                        }}
                      >
                        <div className="stats-record-head">
                          <strong>{record.file_name}</strong>
                          <span>{formatDateTimeLabel(record.graded_at) || record.graded_at}</span>
                        </div>
                        <div className="stats-record-sub">SBD/MSSV: {String(record.data?.student_id || "-")}</div>
                        <div className="stats-record-sub">Mã đề: {String(record.data?.exam_code || "-")} | Điểm: {record.data?.score ?? "-"}</div>
                        <div className="stats-record-actions">
                          <button
                            type="button"
                            className="mini-btn danger tap-feedback stats-record-delete-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteGradeRecord(record.id);
                            }}
                          >
                            Xóa lịch sử
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="stats-record-detail">
                    {!selectedGradeRecord ? (
                      <div className="empty-box">Chọn một bản ghi ở cột trái để xem chi tiết.</div>
                    ) : (
                      <>
                        <div className="stats-detail-grid">
                          <div className="stats-detail-cell">
                            <span>Số báo danh</span>
                            <strong>{String(selectedGradeRecord.data?.student_id || "-")}</strong>
                          </div>
                          <div className="stats-detail-cell">
                            <span>Mã đề</span>
                            <strong>{String(selectedGradeRecord.data?.exam_code || "-")}</strong>
                          </div>
                          <div className="stats-detail-cell">
                            <span>MSSV</span>
                            <strong>{String(selectedGradeRecord.data?.student_id || "-")}</strong>
                          </div>
                          <div className="stats-detail-cell">
                            <span>Nguồn ảnh</span>
                            <strong>{selectedGradeRecord.source === "batch" ? "Batch" : "Camera/Single"}</strong>
                          </div>
                        </div>

                        <div className="stats-image-grid">
                          <button
                            type="button"
                            className={`stats-image-card tap-feedback ${selectedGradeRecord.sid_crop_url ? "" : "disabled"}`}
                            onClick={() => {
                              if (selectedGradeRecord.sid_crop_url) setViewImage(selectedGradeRecord.sid_crop_url);
                            }}
                            disabled={!selectedGradeRecord.sid_crop_url}
                          >
                            {selectedGradeRecord.sid_crop_url ? <img src={selectedGradeRecord.sid_crop_url} alt="SID crop" /> : <div className="stats-image-empty">Không có ảnh SID</div>}
                            <span>SID crop</span>
                          </button>

                          <button
                            type="button"
                            className={`stats-image-card tap-feedback ${selectedGradeRecord.mcq_crop_url ? "" : "disabled"}`}
                            onClick={() => {
                              if (selectedGradeRecord.mcq_crop_url) setViewImage(selectedGradeRecord.mcq_crop_url);
                            }}
                            disabled={!selectedGradeRecord.mcq_crop_url}
                          >
                            {selectedGradeRecord.mcq_crop_url ? <img src={selectedGradeRecord.mcq_crop_url} alt="MCQ crop" /> : <div className="stats-image-empty">Không có ảnh MCQ</div>}
                            <span>MCQ crop</span>
                          </button>

                          <button
                            type="button"
                            className={`stats-image-card tap-feedback ${selectedGradeRecord.image_url ? "" : "disabled"}`}
                            onClick={() => {
                              if (selectedGradeRecord.image_url) setViewImage(selectedGradeRecord.image_url);
                            }}
                            disabled={!selectedGradeRecord.image_url}
                          >
                            {selectedGradeRecord.image_url ? <img src={selectedGradeRecord.image_url} alt="Kết quả chấm" /> : <div className="stats-image-empty">Không có ảnh kết quả</div>}
                            <span>Ảnh kết quả</span>
                          </button>
                        </div>

                        <div className="stats-telemetry-actions">
                          <button
                            type="button"
                            className={`ghost-btn tap-feedback stats-telemetry-btn ${selectedTelemetryJsonUrl ? "" : "disabled"}`}
                            onClick={() => {
                              if (selectedTelemetryJsonUrl) {
                                window.open(selectedTelemetryJsonUrl, "_blank", "noopener,noreferrer");
                              }
                            }}
                            disabled={!selectedTelemetryJsonUrl}
                          >
                            Xem telemetry JSON
                          </button>

                          {selectedTelemetryJsonUrl ? (
                            <a
                              className="mini-btn tap-feedback template-download-btn stats-telemetry-link"
                              href={selectedTelemetryJsonUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              download
                            >
                              Tải telemetry JSON
                            </a>
                          ) : (
                            <div className="stats-telemetry-missing">Chưa có telemetry JSON cho bản ghi này.</div>
                          )}
                        </div>

                        <div className="stats-json-wrap">
                          <div className="stats-json-title">Kết quả trích xuất</div>
                          <pre>{JSON.stringify(selectedGradeRecord.data, null, 2)}</pre>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            {detailTab === "export" && (
              <div className="export-wrap">
                <button className="export-btn tap-feedback" onClick={exportPdfSummary}>📄 Xuất PDF</button>
                <button className="export-btn tap-feedback" onClick={exportExcel}>📊 Xuất Excel</button>
              </div>
            )}
          </section>
        )}
      </main>

      {!showDetail && navTab === "tests" && (
        <button className="fab tap-feedback" onClick={() => setSheetOpen(true)}>
          +
        </button>
      )}

      <nav className="bottom-nav">
        <button
          className={`nav-btn tap-feedback ${navTab === "home" ? "active" : ""}`}
          onClick={() => {
            setDetailTestId(null);
            setNavTab("home");
            stopCamera();
          }}
        >
          Trang chủ
        </button>
        <button
          className={`nav-btn tap-feedback ${navTab === "tests" ? "active" : ""}`}
          onClick={() => {
            setDetailTestId(null);
            setNavTab("tests");
            stopCamera();
          }}
        >
          Bài thi
        </button>
        <button
          className={`nav-btn tap-feedback ${navTab === "templates" ? "active" : ""}`}
          onClick={() => {
            setDetailTestId(null);
            setNavTab("templates");
            stopCamera();
          }}
        >
          Mẫu có sẵn
        </button>
      </nav>

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
