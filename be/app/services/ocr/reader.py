from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import numpy as np
import torch
import os
import io
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

prompt_text = """Bạn là một chuyên gia nhận dạng chữ viết tay tiếng Việt (OCR). 
Nhiệm vụ của bạn là trích xuất toàn bộ nội dung trong bức ảnh này.

Hãy tuân thủ nghiêm ngặt các quy tắc sau:
1. Đọc chính xác nội dung, giữ nguyên các dấu câu và ngắt dòng (line breaks) như trong ảnh.
2. TỰ ĐỘNG KHÔI PHỤC các từ viết tắt thông dụng (ví dụ: 'n' thành 'người', kí hiệu 'o' có gạch dưới thành 'không', 'CSDL' thành 'cơ sở dữ liệu').
3. TUYỆT ĐỐI KHÔNG giải thích, không thêm lời chào, chỉ trả về duy nhất nội dung văn bản.
"""

def _to_pil(src) -> Image.Image:
    """Convert various image sources (path, numpy array, PIL Image) to PIL RGB Image."""
    if isinstance(src, Image.Image):
        return src.convert('RGB')
    if isinstance(src, np.ndarray):
        return Image.fromarray(src).convert('RGB')
    if isinstance(src, str):
        return Image.open(src).convert('RGB')
    raise ValueError(f"Unsupported image type: {type(src)}")


def _to_png_data_url(src) -> str:
    """Convert image source to PNG data URL for OpenAI vision API."""
    if isinstance(src, np.ndarray):
        arr = src
        # OpenCV images are usually BGR; convert to RGB before PNG encoding.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, ::-1]
        img = Image.fromarray(arr).convert('RGB')
    else:
        img = _to_pil(src)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


class OCRReader2:
    def __init__(self, gpu=False, model_type='handwritten'):
        self.device = 'cuda:0' if gpu and torch.cuda.is_available() else 'cpu'
        
        # --- CẤU HÌNH ĐỘNG DỰA TRÊN LOẠI CHỮ ---
        if model_type == 'printed':
            # CHIẾN LƯỢC CHO CHỮ IN: Dùng model vgg_seq2seq (nhẹ và nhanh)
            print(f"   [OCR] Loading 'vgg_seq2seq' for Printed text (GPU={gpu})...")
            try:
                self.config = Cfg.load_config_from_name('vgg_seq2seq')
            except:
                # Fallback nếu không tải được
                print("   [OCR] Fallback to 'vgg_transformer' for Printed text.")
                self.config = Cfg.load_config_from_name('vgg_transformer')
            
            # Tắt beamsearch để tăng tốc độ tối đa cho chữ in
            self.config['predictor']['beamsearch'] = False 
        else:
            # CHIẾN LƯỢC CHO CHỮ VIẾT TAY: Dùng model vgg_transformer (Chính xác cao)
            print(f"   [OCR] Loading 'vgg_transformer' for Handwritten text (GPU={gpu})...")
            self.config = Cfg.load_config_from_name('vgg_transformer')
            
            # Beamsearch giúp đọc chữ viết tay tốt hơn (nhưng chậm hơn chút)
            self.config['predictor']['beamsearch'] = True 

        # Cấu hình thiết bị chung
        self.config['device'] = self.device
        
        # Khởi tạo Predictor
        self.detector = Predictor(self.config)

    def predict(self, img_source):
        """
        Hàm mới: Chuyên dùng để dự đoán text từ 1 ảnh crop
        Input: đường dẫn ảnh (str) HOẶC ảnh numpy array HOẶC PIL Image
        Output: Chuỗi text (str)
        """
        try:
            # Xử lý input đa dạng
            if isinstance(img_source, str):
                img = Image.open(img_source).convert('RGB')
            elif isinstance(img_source, np.ndarray):
                img = Image.fromarray(img_source).convert('RGB')
            elif isinstance(img_source, Image.Image):
                img = img_source.convert('RGB')
            else:
                return "" # Input không hợp lệ

            # Dự đoán
            return self.detector.predict(img)
        except Exception as e:
            print(f"Error in OCR predict: {e}")
            return ""

    def read(self, img_path):
        """
        Hàm cũ: Giữ lại để tương thích với code cũ (trả về list tuple)
        """
        text = self.predict(img_path)
        # VietOCR không có bbox & confidence → trả theo format giả lập
        return [(None, text, 1.0)]


class OCRReaderInternVL:
    """
    InternVL2 — Vision-Language model mạnh cho OCR đa ngôn ngữ (kể cả tiếng Việt).
    Mặc định dùng InternVL2-2B (nhỏ nhất, ~4 GB).
    Tải từ HuggingFace lần đầu.
    """
    def __init__(self, model_name='OpenGVLab/InternVL2-2B', gpu=False):
        from transformers import AutoModel, AutoTokenizer

        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if gpu and torch.cuda.is_available() else torch.float32
        print(f"   [OCR] Loading InternVL2 model: {model_name} (device={self.device})...")
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("   [OCR] [OK] InternVL2 loaded")

    def predict(self, img_source) -> str:
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode

            img = _to_pil(img_source)

            # InternVL2 preprocessing
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD  = (0.229, 0.224, 0.225)
            transform = T.Compose([
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            pixel_values = transform(img).unsqueeze(0).to(
                next(self.model.parameters()).dtype
            ).to(self.device)

            generation_config = dict(max_new_tokens=512, do_sample=False)
            question = '<image>\n'+ prompt_text
            response = self.model.chat(
                self.tokenizer, pixel_values, question, generation_config
            )
            return response.strip()
        except Exception as e:
            print(f"InternVL2 predict error: {e}")
            return ""


class OCRReaderOpenAI4oMini:
    """
    OpenAI GPT-4o mini OCR via Vision API.
    Requires OPENAI_API_KEY in .env file: be/.env
    """

    def __init__(self, model_name='gpt-4o-mini'):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Thiếu package 'openai'. Chạy: pip install openai")

        api_key = os.getenv('OPENAI_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError("Thiếu OPENAI_API_KEY trong .env. Hãy thêm key vào file be/.env")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"   [OCR] OpenAI vision ready (model={self.model_name})")

    def predict(self, img_source) -> str:
        try:
            image_data_url = _to_png_data_url(img_source)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url, "detail": "high"},
                            },
                        ],
                    }
                ],
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(f"OpenAI GPT-4o mini predict error: {e}")
            return ""


