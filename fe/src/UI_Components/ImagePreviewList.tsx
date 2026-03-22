import { X } from "lucide-react";

export default function ImagePreviewList({ images, onRemove, onView }: any) {
    return images.map((img: string, idx: number) => (
        <div className="preview-item" key={idx}>
            <img
                src={img}
                className="preview-thumb"
                onClick={() => onView(img)}
            />
            <button className="delete-btn" onClick={() => onRemove(idx)}>
                <X size={16} />
            </button>
        </div>
    ));
}
