import { X } from "lucide-react";

type ImagePreviewListProps = {
    images: string[];
    onRemove: (index: number) => void;
    onView: (img: string) => void;
};

export default function ImagePreviewList({ images, onRemove, onView }: ImagePreviewListProps) {
    return images.map((img: string, idx: number) => (
        <div className="preview-item" key={idx}>
            <img
                src={img}
                className="preview-thumb"
                alt={`Bai lam ${idx + 1}`}
                onClick={() => onView(img)}
            />
            <button type="button" className="delete-btn" onClick={() => onRemove(idx)} aria-label={`Xoa bai lam ${idx + 1}`}>
                <X size={16} />
            </button>
        </div>
    ));
}
