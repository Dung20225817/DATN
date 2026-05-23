import { API_CONFIG } from "../../config/api";
import { MAX_GRADE_RECORDS, OPTION_LABELS } from "./types";
import type {
  AnswerCompareItem,
  AnswerOverlayBox,
  BubbleTelemetry,
  ExportRecordRow,
  GradeRecord,
  HandwritingRoiOverlay,
  LastResultPayload,
  OMRResult,
  RoiRect,
} from "./types";

export const clampNumber = (value: number, min: number, max: number) => {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
};

export const isRecordObject = (value: unknown): value is Record<string, unknown> => {
  return !!value && typeof value === "object" && !Array.isArray(value);
};

export const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

export const parseRoiRect = (value: unknown): RoiRect | null => {
  if (!isRecordObject(value)) return null;
  const x = toFiniteNumber(value.x);
  const y = toFiniteNumber(value.y);
  const w = toFiniteNumber(value.w);
  const h = toFiniteNumber(value.h);
  if (x == null || y == null || w == null || h == null) return null;
  if (w <= 0 || h <= 0) return null;
  return { x, y, w, h };
};

export const mergeRoiRects = (rects: Array<RoiRect | null>): RoiRect | null => {
  const validRects = rects.filter((rect): rect is RoiRect => rect !== null);
  if (validRects.length === 0) return null;

  const x1 = Math.min(...validRects.map((rect) => rect.x));
  const y1 = Math.min(...validRects.map((rect) => rect.y));
  const x2 = Math.max(...validRects.map((rect) => rect.x + rect.w));
  const y2 = Math.max(...validRects.map((rect) => rect.y + rect.h));
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) return null;
  return { x: x1, y: y1, w, h };
};

export const resolveHoTenRoi = (mapCandidate: unknown): RoiRect | null => {
  if (!isRecordObject(mapCandidate)) return null;
  const direct = parseRoiRect(mapCandidate.ho_ten);
  if (direct) return direct;
  return mergeRoiRects([parseRoiRect(mapCandidate.ho_ten_1), parseRoiRect(mapCandidate.ho_ten_2)]);
};

export const getHandwritingOverlayRois = (result: OMRResult | null | undefined): HandwritingRoiOverlay[] => {
  if (!result || typeof result !== "object") return [];

  const root = result as Record<string, unknown>;
  const candidateMaps: unknown[] = [];

  const fromRoiBoxes = [root.roi_boxes, root.roi_boxes_refined, root.roi_boxes_json];
  fromRoiBoxes.forEach((candidate) => {
    if (!isRecordObject(candidate)) return;
    candidateMaps.push(candidate.handwriting);
  });

  if (isRecordObject(root.handwriting)) {
    candidateMaps.push(root.handwriting.field_rois);
  }

  for (const mapCandidate of candidateMaps) {
    if (!isRecordObject(mapCandidate)) continue;
    const rect = resolveHoTenRoi(mapCandidate);
    const overlays: HandwritingRoiOverlay[] = rect ? [{ key: "ho_ten", rect }] : [];

    if (overlays.length > 0) {
      return overlays;
    }
  }

  return [];
};

export const optionLabelToIndex = (value: unknown) => {
  if (typeof value !== "string") return -1;
  const normalized = value.trim().toUpperCase();
  return OPTION_LABELS.findIndex((label) => label === normalized);
};

export const toChoiceIndex = (value: unknown, fallbackLabel?: unknown) => {
  if (typeof value === "number" && Number.isInteger(value)) return value;
  if (typeof value === "string" && /^-?\d+$/.test(value.trim())) return Number(value.trim());
  return optionLabelToIndex(fallbackLabel);
};

export const parseCellBoxRect = (value: unknown): RoiRect | null => {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const [x1Raw, y1Raw, x2Raw, y2Raw] = value;
  const x1 = toFiniteNumber(x1Raw);
  const y1 = toFiniteNumber(y1Raw);
  const x2 = toFiniteNumber(x2Raw);
  const y2 = toFiniteNumber(y2Raw);
  if (x1 == null || y1 == null || x2 == null || y2 == null) return null;
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) return null;
  return { x: x1, y: y1, w, h };
};

export const buildAnswerOverlayBoxes = (
  telemetry: BubbleTelemetry | null,
  answerCompare: AnswerCompareItem[] | null | undefined
): AnswerOverlayBox[] => {
  const rows = Array.isArray(telemetry?.rows) ? telemetry.rows : [];
  const compareRows = Array.isArray(answerCompare) ? answerCompare : [];
  if (!rows.length || !compareRows.length) return [];

  const rowByQuestion = new Map<number, { cell_boxes?: unknown }>();
  rows.forEach((row, index) => {
    const question = Number(row?.question ?? index + 1);
    if (Number.isInteger(question) && question > 0) {
      rowByQuestion.set(question, row);
    }
  });

  const overlays: AnswerOverlayBox[] = [];
  const pushOverlay = (
    question: number,
    optionIndex: number,
    kind: AnswerOverlayBox["kind"],
    cellBoxes: unknown
  ) => {
    if (!Array.isArray(cellBoxes) || optionIndex < 0 || optionIndex >= cellBoxes.length) return;
    const rect = parseCellBoxRect(cellBoxes[optionIndex]);
    if (!rect) return;
    overlays.push({
      key: `q${question}_${kind}_${optionIndex}_${overlays.length}`,
      kind,
      question,
      label: OPTION_LABELS[optionIndex] || String(optionIndex + 1),
      rect,
    });
  };

  compareRows.forEach((item, index) => {
    const question = Number(item?.question ?? index + 1);
    if (!Number.isInteger(question) || question <= 0) return;
    const telemetryRow = rowByQuestion.get(question);
    const cellBoxes = telemetryRow?.cell_boxes;
    const selected = toChoiceIndex(item?.selected, item?.selected_label);
    const correct = toChoiceIndex(item?.correct, item?.correct_label);
    const status = String(item?.status || "").trim().toLowerCase();

    if (status === "correct") {
      pushOverlay(question, selected >= 0 ? selected : correct, "correct", cellBoxes);
      return;
    }

    if (status === "wrong") {
      pushOverlay(question, selected, "wrong", cellBoxes);
      if (correct >= 0 && correct !== selected) {
        pushOverlay(question, correct, "missing", cellBoxes);
      }
      return;
    }

    if (status === "uncertain") {
      pushOverlay(question, correct, "missing", cellBoxes);
      return;
    }

    if (status === "no-key") {
      pushOverlay(question, selected, "no-key", cellBoxes);
    }
  });

  return overlays;
};

export function mapClientPointToVideo(
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

export function darkRatioInArea(
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

export function avgLumaInArea(
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

export const nowDateTimeLocal = () => {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
};

export const formatDateTimeLabel = (value: string) => {
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

export const toAnswerString = (answers: string[]) => {
  const oneBased = answers.map((x) => {
    const idx = OPTION_LABELS.indexOf(x);
    return idx >= 0 ? String(idx + 1) : "";
  });
  return oneBased.join(",");
};

export const toAbsoluteStaticUrl = (url: string | null | undefined) => {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  return `${API_CONFIG.BASE_URL}${url}`;
};

export const toStaticOmrUrlFromFileName = (fileName: unknown) => {
  if (typeof fileName !== "string") return null;
  const normalizedPath = fileName.trim().replaceAll("\\", "/");
  const safeName = normalizedPath.split("/").pop()?.trim() || "";
  if (!safeName) return null;
  return toAbsoluteStaticUrl(`/static/omr/${encodeURIComponent(safeName)}`);
};

export const toHandwritingCropUrl = (value: unknown) => {
  if (typeof value !== "string") return null;
  const safeValue = value.trim();
  if (!safeValue) return null;
  if (/^https?:\/\//i.test(safeValue)) return safeValue;
  if (safeValue.startsWith("/")) return toAbsoluteStaticUrl(safeValue);
  return toStaticOmrUrlFromFileName(safeValue);
};

export const getHoTenCropUrl = (record: GradeRecord | null | undefined) => {
  if (!record || !isRecordObject(record.data)) return null;

  const directUrl = toHandwritingCropUrl(record.data.ho_ten_crop_url);
  if (directUrl) return directUrl;

  const handwriting = record.data.handwriting;
  if (!isRecordObject(handwriting)) return null;

  const cropImagesUrl = handwriting.crop_images_url;
  if (isRecordObject(cropImagesUrl)) {
    const hoTenUrl = toHandwritingCropUrl(cropImagesUrl.ho_ten);
    if (hoTenUrl) return hoTenUrl;
  }

  const cropImages = handwriting.crop_images;
  if (isRecordObject(cropImages)) {
    const hoTenFromCrop = toHandwritingCropUrl(cropImages.ho_ten);
    if (hoTenFromCrop) return hoTenFromCrop;
  }

  const preprocessedCropImages = handwriting.preprocessed_crop_images;
  if (!isRecordObject(preprocessedCropImages)) return null;
  return toHandwritingCropUrl(preprocessedCropImages.ho_ten);
};

export const createRecordId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
};

export const normalizeRecordUrls = (record: GradeRecord): GradeRecord => {
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

export const buildLegacyRecordFromLastResult = (
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

export const getGradeRecordsFromLastResult = (lastResult: LastResultPayload | OMRResult | null | undefined): GradeRecord[] => {
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

export const buildGradeRecord = ({
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

export const mergeLastResultWithRecords = (
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

export const rebuildLastResultFromRecords = (
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

export const buildExportRecordRows = (records: GradeRecord[]): ExportRecordRow[] => {
  return records.map((record) => ({
    mssv: String(record.data?.student_id ?? "-").trim() || "-",
    examCode: String(record.data?.exam_code ?? "-").trim() || "-",
    score: String(record.data?.score ?? "-").trim() || "-",
  }));
};

export const toSafeFileName = (value: string) => {
  const normalized = value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "_");
  return normalized || "omr_export";
};

export function downloadBlobFile(fileName: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
