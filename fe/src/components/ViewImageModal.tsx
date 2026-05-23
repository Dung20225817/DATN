type ViewImageModalProps = {
    img: string;
    onClose: () => void;
};

export default function ViewImageModal({ img, onClose }: ViewImageModalProps) {
    return (
        <div className="view-overlay" onClick={onClose}>
            <img src={img} className="view-large" alt="Anh bai lam phong to" />
        </div>
    );
}
