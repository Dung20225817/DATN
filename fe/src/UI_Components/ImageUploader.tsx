import { Plus } from "lucide-react";

export default function ImageUploader({ onClick }: { onClick: () => void }) {
    return (
        <div
            className="upload-box"
            onClick={onClick}
            onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onClick();
                }
            }}
            role="button"
            tabIndex={0}
            aria-label="Tải ảnh bài làm"
        >
            <Plus className="plus" size={48} strokeWidth={3} />
        </div>
    );
}
