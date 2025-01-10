from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download
import torch
import os

class Predictor(BasePredictor):
    def setup(self):
        """โหลดโมเดล KhanomTan-TTS"""
        # ดึงโมเดลจาก Hugging Face
        model_path = snapshot_download(repo_id="wannaphong/khanomtan-tts-v1.0")
        model_file = os.path.join(model_path, "best_model.pth")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # โหลดโมเดลด้วย PyTorch
        self.model = torch.load(model_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def predict(
        self,
        text: str = Input(description="ข้อความที่ต้องการแปลงเป็นเสียง"),
        speaker: Path = Input(description="ไฟล์เสียงของผู้พูด (ถ้ามี)", default=None),
        language: str = Input(description="ภาษา (ไทยหรืออังกฤษ)", choices=["th", "en"], default="th"),
    ) -> Path:
        """แปลงข้อความเป็นเสียง"""
        output_path = "/tmp/output.wav"

        # ถ้ามีไฟล์ speaker ให้เตรียมไฟล์
        if speaker:
            speaker_wav = "/tmp/speaker.wav"
            os.system(f"ffmpeg -i {speaker} -ar 22050 -ac 1 -y {speaker_wav}")
        else:
            speaker_wav = None

        # เรียกฟังก์ชัน text_to_speech
        self.model.text_to_speech(text=text, output_path=output_path, speaker_wav=speaker_wav, language=language)
        return Path(output_path)
