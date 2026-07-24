import time
import io
import json
import torch
import numpy as np
import av
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

class GoogleVeoVertexVideoGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": ([
                    "global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1",
                    "us-west1", "us-west2", "us-west3", "us-west4",
                    "northamerica-northeast1", "northamerica-northeast2",
                    "southamerica-east1", "southamerica-west1", "africa-south1",
                    "europe-west1", "europe-north1", "europe-west2", "europe-west3",
                    "europe-west4", "europe-west6", "europe-west8", "europe-west9",
                    "europe-west12", "europe-southwest1", "europe-central2",
                    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2",
                    "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1",
                    "asia-southeast2", "australia-southeast1", "australia-southeast2",
                    "me-central1", "me-central2", "me-west1"
                ], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "veo-3.1-generate-001",
                    "veo-3.1-fast-generate-001",
                    "veo-3.1-generate-preview",
                    "veo-3.1-fast-generate-preview",
                    "veo-3.1-lite-generate-preview"
                ], {"default": "veo-3.1-generate-001"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 4, "min": 4, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "fps": (["24"], {"default": "24"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, False)

    def setup_client(self, service_account_json, project_id, location):
        if not service_account_json.strip():
            raise ValueError("Service account JSON content is required.")
        if not project_id.strip():
            raise ValueError("Project ID is required.")

        try:
            sa_info = json.loads(service_account_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")

        credentials = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        return genai.Client(
            vertexai=True,
            project=project_id.strip(),
            location=location.strip(),
            credentials=credentials,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )

    def generate_video(self, prompt, project_id, location, service_account, model, resolution, aspect_ratio,
                       duration_seconds, seed, generate_audio, fps, negative_prompt=None,
                       first_frame=None, last_frame=None):

        client = self.setup_client(service_account, project_id, location)

        def tensor_to_bytes(t):
            if t.dim() == 4: t = t[0]
            arr = (t.cpu().numpy() * 255).astype(np.uint8)
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format="PNG")
            return b.getvalue()

        config = types.GenerateVideosConfig(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            generate_audio=generate_audio,
            fps=int(fps),
            seed=seed if seed != -1 else None,
            negative_prompt=negative_prompt.strip() if negative_prompt and negative_prompt.strip() else None
        )

        gen_kwargs = {"model": model, "prompt": prompt, "config": config}

        if first_frame is not None:
            gen_kwargs["image"] = types.Image(image_bytes=tensor_to_bytes(first_frame), mime_type="image/png")
        if last_frame is not None:
            setattr(config, 'last_frame', types.Image(image_bytes=tensor_to_bytes(last_frame), mime_type="image/png"))

        op = client.models.generate_videos(**gen_kwargs)
        print(f"Veo Operation: {op.name}")

        while not op.done:
            time.sleep(5)
            op = client.operations.get(op)

        if op.error: raise Exception(f"Veo Error: {op.error}")
        if not op.result.generated_videos: raise Exception("No videos generated")

        video_bytes = io.BytesIO(op.result.generated_videos[0].video.video_bytes)

        container = av.open(video_bytes)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).unsqueeze(0))
        container.close()

        audio = None
        if generate_audio:
            video_bytes.seek(0)
            container = av.open(video_bytes)
            if container.streams.audio:
                audio_data = [f.to_ndarray() for f in container.decode(audio=0)]
                if audio_data:
                    waveform = torch.from_numpy(np.concatenate(audio_data, axis=1)).float()
                    if audio_data[0].dtype == np.int16: waveform /= 32768.0
                    elif audio_data[0].dtype == np.int32: waveform /= 2147483648.0
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": container.streams.audio[0].rate}
            container.close()

        if not frames: raise Exception("Failed to decode video frames")

        return ([torch.cat(frames, dim=0)], audio)


NODE_CLASS_MAPPINGS = {"GoogleVeoVertexVideoGenerator": GoogleVeoVertexVideoGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleVeoVertexVideoGenerator": "Google Veo (Vertex AI)"}