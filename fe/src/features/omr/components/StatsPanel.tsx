import { useEffect, useMemo, useState } from "react";
import type { AnswerOverlayBox, BubbleTelemetry, GradeRecord, HandwritingRoiOverlay } from "../types";
import { OMR_CANVAS_HEIGHT, OMR_CANVAS_WIDTH } from "../types";
import {
  buildAnswerOverlayBoxes,
  clampNumber,
  formatDateTimeLabel,
  getHoTenCropUrl,
} from "../utils";

type StatsPanelProps = {
  selectionMode: "inline" | "route";
  filteredGradeRecords: GradeRecord[];
  gradeRecords: GradeRecord[];
  statsFilterMode: "exam_code" | "student_id";
  statsFilterValue: string;
  setStatsFilterMode: (value: "exam_code" | "student_id") => void;
  setStatsFilterValue: (value: string) => void;
  selectedGradeRecord: GradeRecord | null;
  handwritingOverlayRois: HandwritingRoiOverlay[];
  onOpenRecordDetail: (record: GradeRecord) => void;
  onDeleteGradeRecord: (recordId: string) => void;
  onViewImage: (url: string) => void;
};

type TelemetryState = {
  url: string | null;
  status: "ready" | "error";
  data: BubbleTelemetry | null;
};

function toOverlayStyle(rect: { x: number; y: number; w: number; h: number }) {
  const left = clampNumber((rect.x / OMR_CANVAS_WIDTH) * 100, 0, 100);
  const top = clampNumber((rect.y / OMR_CANVAS_HEIGHT) * 100, 0, 100);
  const width = clampNumber((rect.w / OMR_CANVAS_WIDTH) * 100, 0, 100 - left);
  const height = clampNumber((rect.h / OMR_CANVAS_HEIGHT) * 100, 0, 100 - top);
  return { left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` };
}

function renderAnswerOverlay(box: AnswerOverlayBox) {
  return (
    <span
      key={box.key}
      className={`stats-answer-box stats-answer-box--${box.kind}`}
      style={toOverlayStyle(box.rect)}
      title={`Câu ${box.question}: ${box.label}`}
    />
  );
}

export default function StatsPanel({
  selectionMode,
  filteredGradeRecords,
  gradeRecords,
  statsFilterMode,
  statsFilterValue,
  setStatsFilterMode,
  setStatsFilterValue,
  selectedGradeRecord,
  handwritingOverlayRois,
  onOpenRecordDetail,
  onDeleteGradeRecord,
  onViewImage,
}: StatsPanelProps) {
  const isRouteDetail = selectionMode === "route";
  const telemetryUrl = isRouteDetail ? (selectedGradeRecord?.bubble_confidence_json_url || null) : null;
  const [telemetryState, setTelemetryState] = useState<TelemetryState>({ url: null, status: "error", data: null });
  const [resultPreviewOpen, setResultPreviewOpen] = useState(false);

  useEffect(() => {
    if (!telemetryUrl) return;

    const controller = new AbortController();

    fetch(telemetryUrl, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Không tải được telemetry (${res.status})`);
        return res.json() as Promise<BubbleTelemetry>;
      })
      .then((data) => setTelemetryState({ url: telemetryUrl, status: "ready", data }))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setTelemetryState({ url: telemetryUrl, status: "error", data: null });
      });

    return () => controller.abort();
  }, [telemetryUrl]);

  const activeTelemetryData =
    telemetryUrl && telemetryState.url === telemetryUrl && telemetryState.status === "ready"
      ? telemetryState.data
      : null;
  const telemetryDisplayStatus =
    !telemetryUrl
      ? "idle"
      : telemetryState.url !== telemetryUrl
        ? "loading"
        : telemetryState.status;

  const answerOverlayBoxes = useMemo(
    () => buildAnswerOverlayBoxes(activeTelemetryData, selectedGradeRecord?.data?.answer_compare),
    [activeTelemetryData, selectedGradeRecord?.data?.answer_compare]
  );

  const renderResultOverlay = (imageUrl: string, variant: "inline" | "large") => (
    <div className={`stats-result-overlay-wrap ${variant === "large" ? "large" : ""}`}>
      <img src={imageUrl} alt="Ảnh kết quả chấm" />
      {answerOverlayBoxes.map(renderAnswerOverlay)}
      {handwritingOverlayRois.map(({ key, rect }) => (
        <span
          key={`hw_roi_${key}`}
          className="stats-roi-box handwriting"
          title="ROI họ và tên"
          style={toOverlayStyle(rect)}
        >
          <small>Họ tên</small>
        </span>
      ))}
    </div>
  );

  if (!isRouteDetail) {
    return (
      <div className="stats-wrap">
        <div className="stats-record-list stats-record-list-compact">
          <div className="stats-record-header">
            <div className="stats-record-title">Danh sách các bài đã chấm ({filteredGradeRecords.length}/{gradeRecords.length})</div>
          </div>

          <div className="stats-filter-wrap">
            <div className="stats-filter-switch-row">
              <button
                type="button"
                className={`stats-filter-chip tap-feedback ${statsFilterMode === "exam_code" ? "active" : ""}`}
                onClick={() => setStatsFilterMode("exam_code")}
              >
                Lọc theo Mã đề
              </button>
              <button
                type="button"
                className={`stats-filter-chip tap-feedback ${statsFilterMode === "student_id" ? "active" : ""}`}
                onClick={() => setStatsFilterMode("student_id")}
              >
                Lọc theo MSSV
              </button>
            </div>

            <div className="stats-filter-input-row">
              <input
                className="stats-filter-input"
                value={statsFilterValue}
                onChange={(e) => setStatsFilterValue(e.target.value.trimStart())}
                placeholder={statsFilterMode === "exam_code" ? "Nhập mã đề cần lọc" : "Nhập MSSV cần lọc"}
              />
              <button
                type="button"
                className="mini-btn tap-feedback stats-filter-reset-btn"
                disabled={!statsFilterValue}
                onClick={() => setStatsFilterValue("")}
              >
                Xóa lọc
              </button>
            </div>
          </div>

          {gradeRecords.length === 0 && (
            <div className="empty-box">Chưa có bản ghi theo từng ảnh. Hãy chấm ít nhất 1 ảnh để tạo danh sách.</div>
          )}

          {gradeRecords.length > 0 && filteredGradeRecords.length === 0 && (
            <div className="empty-box">Không có bản ghi nào khớp bộ lọc hiện tại.</div>
          )}

          {filteredGradeRecords.map((record) => {
            const hoTenCropUrl = getHoTenCropUrl(record);
            return (
              <div
                key={record.id}
                className="stats-record-item tap-feedback"
                onClick={() => onOpenRecordDetail(record)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onOpenRecordDetail(record);
                  }
                }}
              >
                <div className="stats-record-basic-grid">
                  <div className="stats-record-sub"><strong>MSSV:</strong> {String(record.data?.student_id || "-")}</div>
                  <div className="stats-record-sub"><strong>Mã đề:</strong> {String(record.data?.exam_code || "-")}</div>
                  <div className="stats-record-sub"><strong>Điểm:</strong> {record.data?.score ?? "-"}</div>
                  <div className="stats-record-sub"><strong>Thời gian xử lý:</strong> {formatDateTimeLabel(record.graded_at) || record.graded_at}</div>
                </div>

                <div className="stats-name-crop-row">
                  <span className="stats-name-crop-label">Họ và tên</span>
                  {hoTenCropUrl ? (
                    <img className="stats-name-crop-image" src={hoTenCropUrl} alt="Họ tên đã crop" />
                  ) : (
                    <div className="stats-name-crop-empty">Không có ảnh họ tên.</div>
                  )}
                </div>

                <div className="stats-record-actions">
                  <button
                    type="button"
                    className="mini-btn danger tap-feedback stats-record-delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteGradeRecord(record.id);
                    }}
                  >
                    Xóa
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  const hoTenCropUrl = getHoTenCropUrl(selectedGradeRecord);

  return (
    <div className="stats-wrap">
      {!selectedGradeRecord ? (
        <div className="empty-box">Không tìm thấy bản ghi đã chọn.</div>
      ) : (
        <div className="stats-record-detail stats-record-detail-single">
          <div className="stats-detail-grid stats-detail-grid-single">
            <div className="stats-detail-cell">
              <span>MSSV</span>
              <strong>{String(selectedGradeRecord.data?.student_id || "-")}</strong>
            </div>
            <div className="stats-detail-cell">
              <span>Mã đề</span>
              <strong>{String(selectedGradeRecord.data?.exam_code || "-")}</strong>
            </div>
            <div className="stats-detail-cell">
              <span>Điểm</span>
              <strong>{selectedGradeRecord.data?.score ?? "-"}</strong>
            </div>
            <div className="stats-detail-cell">
              <span>Thời gian xử lý</span>
              <strong>{formatDateTimeLabel(selectedGradeRecord.graded_at) || selectedGradeRecord.graded_at}</strong>
            </div>
            <div className="stats-detail-cell">
              <span>Tên file</span>
              <strong>{selectedGradeRecord.file_name || "-"}</strong>
            </div>
          </div>

          <div className="stats-name-crop-row">
            <span className="stats-name-crop-label">Họ và tên</span>
            {hoTenCropUrl ? (
              <img className="stats-name-crop-image" src={hoTenCropUrl} alt="Họ tên đã crop" />
            ) : (
              <div className="stats-name-crop-empty">Không có ảnh họ tên.</div>
            )}
          </div>

          <button
            type="button"
            className={`stats-image-card tap-feedback ${selectedGradeRecord.image_url ? "" : "disabled"}`}
            onClick={() => {
              if (!selectedGradeRecord.image_url) return;
              if (isRouteDetail) {
                setResultPreviewOpen(true);
              } else {
                onViewImage(selectedGradeRecord.image_url);
              }
            }}
            disabled={!selectedGradeRecord.image_url}
          >
            {selectedGradeRecord.image_url ? (
              <>
                {renderResultOverlay(selectedGradeRecord.image_url, "inline")}
                <div className="stats-overlay-legend" aria-hidden="true">
                  <span className="legend-item correct">Đúng</span>
                  <span className="legend-item wrong">Sai</span>
                  <span className="legend-item missing">??p ?n Đúng ch?a t?</span>
                  <span className="legend-item no-key">Không có đáp án</span>
                </div>
                {telemetryDisplayStatus === "loading" && <div className="stats-overlay-note">Đang tải dữ liệu khoanh đáp án...</div>}
                {telemetryDisplayStatus === "error" && <div className="stats-overlay-note">Không tải được dữ liệu khoanh đáp án, ảnh gốc vẫn được hiển thị.</div>}
              </>
            ) : (
              <div className="stats-image-empty">Không có ảnh kết quả</div>
            )}
            <span>Ảnh kết quả</span>
          </button>

          {resultPreviewOpen && selectedGradeRecord.image_url && (
            <div className="view-overlay stats-result-preview-overlay" onClick={() => setResultPreviewOpen(false)}>
              <div className="stats-result-preview-panel" onClick={(event) => event.stopPropagation()}>
                {renderResultOverlay(selectedGradeRecord.image_url, "large")}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
