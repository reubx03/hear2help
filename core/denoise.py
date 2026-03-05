import os
import uuid
import subprocess
import sys


class DenoiseUnit:
    def __init__(self, enabled=True, output_dir="data/denoised"):
    
        self.enabled = enabled
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def denoise(self, audio_path: str) -> str:
        if not self.enabled:
            return audio_path

        try:
            model_name = "mdx_extra_q"   # keep this in ONE place

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "demucs",
                    "-n", model_name,
                    "--two-stems", "vocals",
                    "--out", self.output_dir,
                    audio_path,
                ],
                check=True,
            )

            base = os.path.splitext(os.path.basename(audio_path))[0]

            generated = os.path.join(
                self.output_dir,
                model_name,
                base,
                "vocals.wav"
            )

            if not os.path.exists(generated):
                print("[DENOISE] Expected output not found, fallback")
                return audio_path

            out_path = os.path.join(
                self.output_dir,
                f"denoised_{uuid.uuid4().hex}.wav"
            )

            os.replace(generated, out_path)
            print(f"[DENOISE] Audio enhanced → {out_path}")

            return out_path

        except Exception as e:
            print(f"[DENOISE ERROR] {e}")
            return audio_path
