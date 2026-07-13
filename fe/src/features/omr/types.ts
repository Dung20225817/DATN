export type NavTab = "home" | "templates";
export type DetailTab = "grading" | "answers" | "stats" | "export";
export type ScannerUiState = "idle" | "searching" | "locked";
export type PickedImageSource = "camera" | "library" | "mixed" | null;

export type FormProfile = {
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

export type CornerKey = "tl" | "tr" | "bl" | "br";

export type MarkerBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  cx?: number;
  cy?: number;
};

export type ScannerHint = {
  min_dark_ratio?: number;
  min_center_luma?: number;
  min_marker_contrast?: number;
  sample_size_norm?: number;
};

export type RoiRect = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type HandwritingFieldsStrategy = {
  enabled?: boolean;
  save_crops?: boolean;
  field_rois?: {
    ho_ten?: RoiRect;
    ho_ten_1?: RoiRect;
    ho_ten_2?: RoiRect;
    [key: string]: RoiRect | undefined;
  };
};

export type McqDecodeStrategy = {
  threshold_mode?: "otsu" | "adaptive";
  row_offsets_px?: number[];
  [key: string]: unknown;
};

export type ScannerStrategy = {
  sheet_aspect_ratio?: number;
  sid_roi?: RoiRect | null;
  mcq_roi?: RoiRect | null;
  exam_code_roi?: RoiRect | null;
  handwriting_fields?: HandwritingFieldsStrategy | null;
  sid_row_offsets?: number[] | null;
  mcq_decode?: McqDecodeStrategy | null;
  disable_mcq_rescue?: boolean;
  sid_roi_lock?: boolean;
  page_size_pt?: { width: number; height: number } | null;
  corner_markers?: Partial<Record<CornerKey, MarkerBox>>;
  scanner_hint?: ScannerHint;
};

export type MarkerSample = {
  x: number;
  y: number;
  sampleSize: number;
};

export type AnswerSet = {
  code: string;
  answers: string[];
};

export type AnswerCompareItem = {
  question?: number;
  selected?: number;
  selected_label?: string;
  correct?: number;
  correct_label?: string;
  status?: string;
  is_correct?: boolean;
};

export type OMRResult = {
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
  ho_ten_crop_url?: string;
  [key: string]: unknown;
};

export type HandwritingRoiOverlay = {
  key: string;
  rect: RoiRect;
};

export type TelemetryRow = {
  question?: number;
  selected?: number;
  cell_boxes?: unknown;
};

export type BubbleTelemetry = {
  rows?: TelemetryRow[];
};

export type AnswerOverlayKind = "correct" | "wrong" | "missing" | "no-key";

export type AnswerOverlayBox = {
  key: string;
  kind: AnswerOverlayKind;
  question: number;
  label: string;
  rect: RoiRect;
};

export type BatchGradeResultItem = {
  file_name?: string;
  success?: boolean;
  data?: OMRResult;
  image_url?: string | null;
  sid_crop_url?: string | null;
  mcq_crop_url?: string | null;
  bubble_confidence_json_url?: string | null;
};

export type GradeRecord = {
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

export type ExportRecordRow = {
  mssv: string;
  examCode: string;
  score: string;
};

export type LastResultPayload = OMRResult & {
  grade_records?: GradeRecord[];
  __meta__?: {
    form_profile_code?: string;
    [key: string]: unknown;
  };
};

export type TestCardItem = {
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

export type AssignmentApiItem = {
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

export const OPTION_LABELS = ["A", "B", "C", "D"];
export const CORNER_KEYS: CornerKey[] = ["tl", "tr", "bl", "br"];
export const MAX_GRADE_RECORDS = 200;
export const AUTO_CAPTURE_COOLDOWN_MS = 1300;
export const OMR_CANVAS_WIDTH = 1000;
export const OMR_CANVAS_HEIGHT = 1400;

export const FALLBACK_MARKER_SAMPLES: Record<CornerKey, MarkerSample> = {
  tl: { x: 0.08, y: 0.08, sampleSize: 0.062 },
  tr: { x: 0.92, y: 0.08, sampleSize: 0.062 },
  bl: { x: 0.08, y: 0.92, sampleSize: 0.062 },
  br: { x: 0.92, y: 0.92, sampleSize: 0.062 },
};
