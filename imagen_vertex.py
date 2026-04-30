import io
import json
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

class GoogleImagenGenerateVertex:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A majestic lion in the savanna"}),
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": (["global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west2", "us-west3", "us-west4", "northamerica-northeast1", "northamerica-northeast2", "southamerica-east1", "southamerica-west1", "africa-south1", "europe-west1", "europe-north1", "europe-west2", "europe-west3", "europe-west4", "europe-west6", "europe-west8", "europe-west9", "europe-west12", "europe-southwest1", "europe-central2", "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1", "asia-southeast2", "australia-southeast1", "australia-southeast2", "me-central1", "me-central2", "me-west1"], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "model": (["imagen-4.0-ultra-generate-001", "imagen-4.0-generate-001", "imagen-4.0-fast-generate-001", "imagen-3.0-generate-002"], {"default": "imagen-4.0-generate-001"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "aspect_ratio": (["1:1", "9:16", "16:9", "4:3", "3:4"], {"default": "1:1"}),
                "image_size": (["1K", "2K"], {"default": "1K"}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "image/generation"

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

    def generate_images(self, prompt, project_id, location, service_account, model, number_of_images, aspect_ratio, image_size, seed, guidance_scale, negative_prompt=""):
        client = self.setup_client(service_account, project_id, location)

        config = types.GenerateImagesConfig(
            number_of_images=number_of_images,
            aspect_ratio=aspect_ratio,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None
        )

        if "imagen-4.0" in model and "fast" not in model:
            config.image_size = image_size

        try:
            result = client.models.generate_images(model=model, prompt=prompt, config=config)
            if not result.generated_images:
                raise ValueError("No images generated")

            tensors = []
            for item in result.generated_images:
                img_data = item.image
                if hasattr(img_data, "image_bytes"):
                    pil_img = Image.open(io.BytesIO(img_data.image_bytes))
                elif hasattr(img_data, "convert"):
                    pil_img = img_data
                else:
                    pil_img = Image.open(io.BytesIO(img_data))

                tensors.append(torch.from_numpy(np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0))

            return (torch.stack(tensors),)

        except Exception as e:
            raise RuntimeError(f"Google Imagen Error: {e}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

NODE_CLASS_MAPPINGS = {"GoogleImagenGenerateVertex": GoogleImagenGenerateVertex}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleImagenGenerateVertex": "Imagen Generate (Vertex AI)"}