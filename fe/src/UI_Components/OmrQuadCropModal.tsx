import { useEffect, useMemo, useRef, useState } from "react";

type CropPoint = { x: number; y: number };

type OmrQuadCropModalProps = {
    imageUrl: string;
    initialPoints: CropPoint[];
    onCancel: () => void;
    onConfirm: (points: CropPoint[]) => void;
};

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

const defaultPoints = (): CropPoint[] => [
    { x: 0.08, y: 0.08 },
    { x: 0.92, y: 0.08 },
    { x: 0.92, y: 0.92 },
    { x: 0.08, y: 0.92 },
];

export default function OmrQuadCropModal({
    imageUrl,
    initialPoints,
    onCancel,
    onConfirm,
}: OmrQuadCropModalProps) {
    const imgRef = useRef<HTMLImageElement | null>(null);
    const [points, setPoints] = useState<CropPoint[]>(defaultPoints());
    const [dragIndex, setDragIndex] = useState<number | null>(null);

    useEffect(() => {
        if (initialPoints.length === 4) {
            setPoints(initialPoints.map((p) => ({ x: clamp01(p.x), y: clamp01(p.y) })));
        } else {
            setPoints(defaultPoints());
        }
    }, [initialPoints]);

    const polygon = useMemo(() => points.map((p) => `${p.x * 100},${p.y * 100}`).join(" "), [points]);

    const pointerToNorm = (clientX: number, clientY: number): CropPoint | null => {
        const img = imgRef.current;
        if (!img) return null;
        const rect = img.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        return {
            x: clamp01((clientX - rect.left) / rect.width),
            y: clamp01((clientY - rect.top) / rect.height),
        };
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (dragIndex === null) return;
        const next = pointerToNorm(e.clientX, e.clientY);
        if (!next) return;
        setPoints((prev) => prev.map((p, i) => (i === dragIndex ? next : p)));
    };

    const handlePointerUp = () => {
        setDragIndex(null);
    };

    return (
        <div className="popup-overlay" onClick={onCancel}>
            <div
                className="popup-box"
                style={{ maxWidth: "920px", width: "95%" }}
                onClick={(e) => e.stopPropagation()}
            >
                <h3>Chinh 4 goc vung cat OMR</h3>
                <p style={{ marginTop: 0, color: "#555", fontSize: "14px" }}>
                    Keo 4 cham tron de canh khop voi 4 goc to de. Ho tro chuot va thao tac cham tren dien thoai.
                </p>

                <div
                    style={{ position: "relative", width: "100%", touchAction: "none", userSelect: "none" }}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerCancel={handlePointerUp}
                    onPointerLeave={handlePointerUp}
                >
                    <img
                        ref={imgRef}
                        src={imageUrl}
                        alt="OMR crop"
                        style={{ width: "100%", maxHeight: "70vh", objectFit: "contain", display: "block", borderRadius: "8px", border: "1px solid #ccc" }}
                        draggable={false}
                    />

                    <svg
                        viewBox="0 0 100 100"
                        preserveAspectRatio="none"
                        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
                    >
                        <polygon
                            points={polygon}
                            fill="rgba(0, 161, 82, 0.15)"
                            stroke="#00a152"
                            strokeWidth="0.45"
                        />
                    </svg>

                    {points.map((p, idx) => (
                        <button
                            key={`handle-${idx}`}
                            type="button"
                            onPointerDown={(e) => {
                                e.preventDefault();
                                setDragIndex(idx);
                            }}
                            style={{
                                position: "absolute",
                                left: `${p.x * 100}%`,
                                top: `${p.y * 100}%`,
                                transform: "translate(-50%, -50%)",
                                width: "24px",
                                height: "24px",
                                borderRadius: "50%",
                                border: "2px solid #ffffff",
                                background: "#ff6f00",
                                boxShadow: "0 0 0 2px rgba(0,0,0,0.25)",
                                cursor: "grab",
                                touchAction: "none",
                                zIndex: 3,
                            }}
                            aria-label={`Corner ${idx + 1}`}
                        />
                    ))}
                </div>

                <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
                    <button
                        className="popup-btn"
                        type="button"
                        onClick={() => setPoints(defaultPoints())}
                    >
                        Dat lai mac dinh
                    </button>
                    <button
                        className="popup-btn"
                        type="button"
                        onClick={() => onConfirm(points)}
                        style={{ background: "#e8f5e9", borderColor: "#2e7d32", color: "#1b5e20" }}
                    >
                        Xac nhan vung cat
                    </button>
                </div>

                <button className="popup-cancel" onClick={onCancel}>
                    Dong
                </button>
            </div>
        </div>
    );
}
