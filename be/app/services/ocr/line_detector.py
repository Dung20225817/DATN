"""
Hybrid Line Detection - Component 2 từ Implementation Plan
Kết hợp ruled line detection và projection-based
"""

import cv2
import numpy as np

class HybridLineDetector:
    """Kết hợp ruled line detection và projection-based"""
    
    def detect_lines(self, image_path: str) -> list:
        """
        Returns: List of cropped line images
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Cannot read image: {image_path}")
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Thử phát hiện ruled lines
        ruled_lines = self._detect_ruled_lines(gray)
        
        if len(ruled_lines) > 3:  # Nếu tìm thấy đủ dòng kẻ
            print(f"✅ Detected {len(ruled_lines)} ruled lines")
            return self._crop_by_ruled_lines(img, ruled_lines)
        else:
            # Fallback 1: Dùng projection
            print(f"⚠️ Ruled lines not found, using projection-based detection")
            crops = self._crop_by_projection(img, gray)
            if crops:
                return crops
            # Fallback 2: Toàn bộ ảnh là 1 khối text
            return self._full_image_fallback(img)
    
    def _detect_ruled_lines(self, gray):
        """Phát hiện dòng kẻ ngang bằng Hough Transform"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # HoughLinesP: Tìm các đoạn thẳng
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180,
            threshold=100,
            minLineLength=200,  # Dòng kẻ phải dài tối thiểu 200px
            maxLineGap=50
        )
        
        if lines is None:
            return []
        
        # Lọc chỉ lấy dòng ngang (góc gần 0° — cho phép tới 15° để xử lý ảnh chụp nghiêng)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15:  # Dòng ngang (góc < 15°)
                horizontal_lines.append((y1 + y2) // 2)  # Lấy tọa độ y trung bình
        
        # Cluster các dòng gần nhau
        horizontal_lines = sorted(set(horizontal_lines))
        clustered = self._cluster_lines(horizontal_lines, threshold=20)
        
        return sorted(clustered)
    
    def _cluster_lines(self, lines, threshold=20):
        """Gộp các dòng gần nhau"""
        if not lines:
            return []
        
        clustered = [lines[0]]
        for line in lines[1:]:
            if line - clustered[-1] > threshold:
                clustered.append(line)
        
        return clustered
    
    def _crop_by_ruled_lines(self, img, y_coords):
        """Crop ảnh theo tọa độ y của ruled lines"""
        crops = []
        for i in range(len(y_coords) - 1):
            y1, y2 = y_coords[i], y_coords[i + 1]
            crop = img[y1:y2, :]
            if crop.shape[0] > 10:  # Bỏ qua dòng quá nhỏ
                crops.append(crop)
        return crops
    
    def _crop_by_projection(self, img, gray):
        """Fallback: Dùng horizontal projection để tìm vùng text"""
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Tính projection (sum theo chiều ngang — số pixel đen trên mỗi hàng)
        h_proj = np.sum(binary, axis=1)
        
        # Ngưỡng: hàng có ít pixel đen hơn threshold → coi là khoảng trống
        threshold = max(np.mean(h_proj) * 0.15, 5)

        # Tìm các TEXT REGION (hàng liên tiếp có nội dung) thay vì valley
        in_text = False
        text_start = 0
        regions = []  # [(y1, y2), ...] — vùng chứa text

        for i, val in enumerate(h_proj):
            if val >= threshold and not in_text:
                text_start = i
                in_text = True
            elif val < threshold and in_text:
                in_text = False
                if i - text_start > 8:  # bỏ những vùng quá nhỏ (noise)
                    regions.append((text_start, i))

        # Đừng bỏ sót vùng text cuối cùng
        if in_text and len(h_proj) - text_start > 8:
            regions.append((text_start, len(h_proj)))

        if not regions:
            return []

        # Gộp các vùng text gần nhau (khoảng gap nhỏ hơn 3px) thành 1 dòng
        merged = [list(regions[0])]
        for y1, y2 in regions[1:]:
            if y1 - merged[-1][1] <= 3:
                merged[-1][1] = y2
            else:
                merged.append([y1, y2])

        # Crop mỗi vùng text (thêm margin 3px mỗi phía)
        crops = []
        h, w = img.shape[:2]
        for y1, y2 in merged:
            y1m = max(0, y1 - 3)
            y2m = min(h, y2 + 3)
            if y2m - y1m > 8:
                crops.append(img[y1m:y2m, :])

        print(f"✅ Projection-based: found {len(crops)} text lines")
        return crops

    def _full_image_fallback(self, img):
        """Last resort: trả về toàn bộ ảnh như 1 dòng duy nhất"""
        print("⚠️ Using full-image fallback — treating whole image as one text block")
        return [img]
