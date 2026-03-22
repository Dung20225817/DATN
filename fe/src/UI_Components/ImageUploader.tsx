import { Plus } from "lucide-react";

export default function ImageUploader({ onClick }: { onClick: () => void }) {
    return (
        <div className="upload-box" onClick={onClick}>
            <Plus className="plus" size={48} strokeWidth={3} />
        </div>
    );
}
