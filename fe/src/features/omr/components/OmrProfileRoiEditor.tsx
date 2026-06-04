import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { API_CONFIG } from "../../../config/api";
import type { FormProfile, MarkerBox, RoiRect } from "../types";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

type RoiPoint = {
  x: number;
  y: number;
};

type OmrProfileRoiEditorProps = {
  profile: FormProfile;
  sampleUrl: string;
  onClose: () => void;
  onSaved: (profile: FormProfile) => void;
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
};

const MIN_ROI_SIZE = 0.005;

type MarkerBounds = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

function clampUnit(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function rectFromPoints(points: RoiPoint[]): RoiRect | null {
  if (points.length < 4) return null;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const x1 = clampUnit(Math.min(...xs));
  const y1 = clampUnit(Math.min(...ys));
  const x2 = clampUnit(Math.max(...xs));
  const y2 = clampUnit(Math.max(...ys));
  const w = x2 - x1;
  const h = y2 - y1;
  if (w < MIN_ROI_SIZE || h < MIN_ROI_SIZE) return null;
  return {
    x: Number(x1.toFixed(6)),
    y: Number(y1.toFixed(6)),
    w: Number(w.toFixed(6)),
    h: Number(h.toFixed(6)),
  };
}

function clampRectToUnit(rect: RoiRect): RoiRect | null {
  const x1 = clampUnit(rect.x);
  const y1 = clampUnit(rect.y);
  const x2 = clampUnit(rect.x + rect.w);
  const y2 = clampUnit(rect.y + rect.h);
  const w = x2 - x1;
  const h = y2 - y1;
  if (w < MIN_ROI_SIZE || h < MIN_ROI_SIZE) return null;
  return {
    x: Number(x1.toFixed(6)),
    y: Number(y1.toFixed(6)),
    w: Number(w.toFixed(6)),
    h: Number(h.toFixed(6)),
  };
}

function pointsFromRect(rect: RoiRect | null | undefined): RoiPoint[] {
  if (!rect || rect.w <= 0 || rect.h <= 0) return [];
  const x1 = clampUnit(rect.x);
  const y1 = clampUnit(rect.y);
  const x2 = clampUnit(rect.x + rect.w);
  const y2 = clampUnit(rect.y + rect.h);
  return [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ];
}

function getHoTenRect(profile: FormProfile): RoiRect | null {
  const rect = profile.strategy?.handwriting_fields?.field_rois?.ho_ten;
  if (!rect || rect.w <= 0 || rect.h <= 0) return null;
  return rect;
}

function markerCenter(marker: MarkerBox | undefined, axis: "x" | "y"): number | null {
  if (!marker) return null;
  const direct = axis === "x" ? marker.cx : marker.cy;
  if (typeof direct === "number" && Number.isFinite(direct)) return direct;
  const base = axis === "x" ? marker.x : marker.y;
  const size = axis === "x" ? marker.w : marker.h;
  if (!Number.isFinite(base) || !Number.isFinite(size)) return null;
  return base + size / 2;
}

function getMarkerBounds(profile: FormProfile): MarkerBounds | null {
  const markers = profile.strategy?.corner_markers;
  if (!markers) return null;

  const tlx = markerCenter(markers.tl, "x");
  const trx = markerCenter(markers.tr, "x");
  const blx = markerCenter(markers.bl, "x");
  const brx = markerCenter(markers.br, "x");
  const tly = markerCenter(markers.tl, "y");
  const try_ = markerCenter(markers.tr, "y");
  const bly = markerCenter(markers.bl, "y");
  const bry = markerCenter(markers.br, "y");
  if ([tlx, trx, blx, brx, tly, try_, bly, bry].some((value) => value == null)) return null;

  const left = ((tlx as number) + (blx as number)) / 2;
  const right = ((trx as number) + (brx as number)) / 2;
  const top = ((tly as number) + (try_ as number)) / 2;
  const bottom = ((bly as number) + (bry as number)) / 2;
  if (right - left < MIN_ROI_SIZE || bottom - top < MIN_ROI_SIZE) return null;
  return { left, right, top, bottom };
}

function pageRectToWarpRect(rect: RoiRect, bounds: MarkerBounds | null): RoiRect {
  if (!bounds) return rect;
  const converted = {
    x: (rect.x - bounds.left) / (bounds.right - bounds.left),
    y: (rect.y - bounds.top) / (bounds.bottom - bounds.top),
    w: rect.w / (bounds.right - bounds.left),
    h: rect.h / (bounds.bottom - bounds.top),
  };
  return clampRectToUnit(converted) || rect;
}

function warpRectToPageRect(rect: RoiRect, bounds: MarkerBounds | null): RoiRect {
  if (!bounds) return rect;
  const converted = {
    x: bounds.left + rect.x * (bounds.right - bounds.left),
    y: bounds.top + rect.y * (bounds.bottom - bounds.top),
    w: rect.w * (bounds.right - bounds.left),
    h: rect.h * (bounds.bottom - bounds.top),
  };
  return clampRectToUnit(converted) || rect;
}

export default function OmrProfileRoiEditor({
  profile,
  sampleUrl,
  onClose,
  onSaved,
  onError,
  onSuccess,
}: OmrProfileRoiEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [points, setPoints] = useState<RoiPoint[]>(() => pointsFromRect(getHoTenRect(profile)));
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [renderError, setRenderError] = useState("");

  const markerBounds = useMemo(() => getMarkerBounds(profile), [profile]);
  const roiRect = useMemo(() => rectFromPoints(points), [points]);
  const savedRoiRect = useMemo(() => (roiRect ? pageRectToWarpRect(roiRect, markerBounds) : null), [markerBounds, roiRect]);
  const isPdf = profile.sample_file.toLowerCase().endsWith(".pdf");

  const drawCanvas = useCallback((nextPoints: RoiPoint[]) => {
    const canvas = canvasRef.current;
    const base = baseCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !base || !ctx) return;

    canvas.width = base.width;
    canvas.height = base.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);

    const rect = rectFromPoints(nextPoints);
    if (rect) {
      const x = rect.x * canvas.width;
      const y = rect.y * canvas.height;
      const w = rect.w * canvas.width;
      const h = rect.h * canvas.height;
      ctx.fillStyle = "rgba(20, 184, 166, 0.16)";
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = Math.max(2, canvas.width / 420);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }

    nextPoints.forEach((point, index) => {
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(5, canvas.width / 140), 0, Math.PI * 2);
      ctx.fillStyle = "#0f766e";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = Math.max(2, canvas.width / 500);
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = `${Math.max(12, canvas.width / 75)}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index + 1), x, y);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function renderSample() {
      setLoading(true);
      setRenderError("");
      try {
        const base = document.createElement("canvas");
        const ctx = base.getContext("2d");
        if (!ctx) throw new Error("Không thể tạo canvas.");

        if (isPdf) {
          const pdf = await pdfjsLib.getDocument(sampleUrl).promise;
          const page = await pdf.getPage(1);
          const viewport = page.getViewport({ scale: 2 });
          base.width = Math.round(viewport.width);
          base.height = Math.round(viewport.height);
          await page.render({ canvas: base, canvasContext: ctx, viewport }).promise;
        } else {
          const image = new Image();
          image.crossOrigin = "anonymous";
          await new Promise<void>((resolve, reject) => {
            image.onload = () => resolve();
            image.onerror = () => reject(new Error("Không thể tải ảnh mẫu."));
            image.src = sampleUrl;
          });
          base.width = image.naturalWidth;
          base.height = image.naturalHeight;
          ctx.drawImage(image, 0, 0);
        }

        if (cancelled) return;
        baseCanvasRef.current = base;
        const initialRect = getHoTenRect(profile);
        const initialPoints = pointsFromRect(initialRect ? warpRectToPageRect(initialRect, markerBounds) : null);
        setPoints(initialPoints);
        requestAnimationFrame(() => drawCanvas(initialPoints));
      } catch (err) {
        if (cancelled) return;
        setRenderError(err instanceof Error ? err.message : "Không thể render phiếu mẫu.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void renderSample();
    return () => {
      cancelled = true;
    };
  }, [drawCanvas, isPdf, markerBounds, profile, sampleUrl]);

  useEffect(() => {
    drawCanvas(points);
  }, [drawCanvas, points]);

  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || loading || saving) return;
    const box = canvas.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) return;

    const x = clampUnit((event.clientX - box.left) / box.width);
    const y = clampUnit((event.clientY - box.top) / box.height);
    setPoints((current) => {
      const seed = current.length >= 4 ? [] : current;
      return [...seed, { x, y }];
    });
  }

  async function saveRoi() {
    if (!savedRoiRect || saving) return;
    setSaving(true);
    try {
      const nextProfile: FormProfile = {
        ...profile,
        strategy: {
          ...(profile.strategy || {}),
          handwriting_fields: {
            ...(profile.strategy?.handwriting_fields || {}),
            enabled: true,
            save_crops: true,
            field_rois: {
              ...(profile.strategy?.handwriting_fields?.field_rois || {}),
              ho_ten: savedRoiRect,
            },
          },
        },
      };

      const res = await fetch(API_CONFIG.OMR.SAVE_FORM_PROFILE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nextProfile),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || "Lưu ROI thất bại.");
      }
      onSaved(payload.profile as FormProfile);
      onSuccess("Đã lưu ROI họ tên cho phiếu mẫu.");
      onClose();
    } catch (err) {
      onError(err instanceof Error ? err.message : "Không thể lưu ROI họ tên.");
    } finally {
      setSaving(false);
    }
  }

  const roiLabel = savedRoiRect
    ? `marker-warp x=${savedRoiRect.x.toFixed(4)}, y=${savedRoiRect.y.toFixed(4)}, w=${savedRoiRect.w.toFixed(4)}, h=${savedRoiRect.h.toFixed(4)}`
    : "Chọn đủ 4 góc vùng họ tên";

  return (
    <div className="sample-preview-overlay" onClick={onClose}>
      <div className="sample-preview-dialog roi-editor-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="sample-preview-head roi-editor-head">
          <div>
            <strong>{profile.title}</strong>
            <div className="roi-editor-sub">ROI họ tên · {profile.sample_file}</div>
          </div>
          <div className="roi-editor-actions">
            <button className="mini-btn tap-feedback" type="button" onClick={() => setPoints([])} disabled={saving}>
              Reset
            </button>
            <button className="primary-btn tap-feedback" type="button" onClick={saveRoi} disabled={!savedRoiRect || saving}>
              {saving ? "Đang lưu..." : "Lưu ROI"}
            </button>
            <button className="header-back-btn tap-feedback" type="button" onClick={onClose}>
              Đóng
            </button>
          </div>
        </div>
        <div className="roi-editor-body">
          <div className="roi-editor-toolbar">
            <span>{points.length}/4 điểm</span>
            <code>{roiLabel}</code>
          </div>
          <div className="roi-editor-canvas-wrap">
            {loading && <div className="empty-box">Đang render phiếu mẫu...</div>}
            {!!renderError && <div className="empty-box">{renderError}</div>}
            <canvas
              ref={canvasRef}
              className={`roi-editor-canvas ${loading || renderError ? "hidden" : ""}`}
              onClick={handleCanvasClick}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
