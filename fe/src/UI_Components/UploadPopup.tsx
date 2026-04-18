type UploadPopupProps = {
    onSelect: (files: FileList | null) => void;
    onClose: () => void;
};

export default function UploadPopup({ onSelect, onClose }: UploadPopupProps) {
    return (
        <div className="popup-overlay" onClick={onClose}>
            <div className="popup-box" onClick={e => e.stopPropagation()}>
                <h3>Chọn nguồn ảnh</h3>

                <label className="popup-btn">
                    📸 Chụp bằng Camera
                    <input
                        type="file"
                        accept="image/*"
                        capture="environment"
                        className="hidden-input"
                        onChange={(e) => onSelect(e.target.files)}
                    />
                </label>

                <label className="popup-btn">
                    🖼 Chọn ảnh có sẵn
                    <input
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden-input"
                        onChange={(e) => onSelect(e.target.files)}
                    />
                </label>

                <button className="popup-cancel" onClick={onClose}>
                    Hủy
                </button>
            </div>
        </div>
    );
}
