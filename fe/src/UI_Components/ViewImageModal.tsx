export default function ViewImageModal({ img, onClose }: any) {
    return (
        <div className="view-overlay" onClick={onClose}>
            <img src={img} className="view-large" />
        </div>
    );
}
