import base64
import io
import json
import re
import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

class GeminiSegmentationVertexNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segment_prompt": ("STRING", {"default": "all objects", "multiline": True}),
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
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash"
                ], {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "thinking_budget": ("INT", {"default": 0, "min": -1, "max": 24576, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_segmentation"
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

    def generate_segmentation(self, image, segment_prompt, project_id, location, service_account, model, temperature, thinking, seed, thinking_budget=0):
        client = self.setup_client(service_account, project_id, location)

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        orig_img = Image.fromarray(img_np)
        orig_w, orig_h = orig_img.size

        scale = min(1024 / orig_w, 1024 / orig_h)
        proc_img = orig_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS) if scale < 1 else orig_img
        pw, ph = proc_img.size

        img_buf = io.BytesIO()
        proc_img.save(img_buf, format='PNG')

        t_config = None
        if "gemini-2.0" not in model.lower():
            budget = thinking_budget if thinking else 0
            if "gemini-2.5-pro" in model.lower() and budget <= 0:
                print("Gemini-2.5-Pro enforces thinking. Defaulting to auto (-1).")
                budget = -1
            t_config = types.ThinkingConfig(thinking_budget=budget)

        prompt = f'Give the segmentation masks for {segment_prompt}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label".'

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(mime_type="image/png", data=img_buf.getvalue()),
                types.Part.from_text(text=prompt)
            ])],
            config=types.GenerateContentConfig(temperature=temperature, seed=seed, thinking_config=t_config)
        )

        try:
            txt = response.text
            if "```json" in txt: txt = re.search(r"```json\n(.*)\n```", txt, re.DOTALL).group(1)
            segments = json.loads(txt)
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {e}")

        final_mask = np.zeros((ph, pw), dtype=np.uint8)

        for seg in segments:
            try:
                ymin, xmin, ymax, xmax = seg['box_2d']
                x1, y1 = int(xmin * pw / 1000), int(ymin * ph / 1000)
                x2, y2 = int(xmax * pw / 1000), int(ymax * ph / 1000)
                w, h = x2 - x1, y2 - y1

                if w <= 0 or h <= 0: continue

                mask_str = seg['mask'].split(",")[1] if "data:image" in seg['mask'] else seg['mask']
                patch = Image.open(io.BytesIO(base64.b64decode(mask_str))).convert('L')

                if patch.size != (w, h):
                    patch = patch.resize((w, h), Image.Resampling.NEAREST)

                patch_arr = np.where(np.array(patch) > 128, 255, 0).astype(np.uint8)

                target_slice = final_mask[y1:y2, x1:x2]
                if target_slice.shape == patch_arr.shape:
                    np.maximum(target_slice, patch_arr, out=target_slice)

            except Exception: continue

        if (pw, ph) != (orig_w, orig_h):
            final_mask = np.array(Image.fromarray(final_mask).resize((orig_w, orig_h), Image.Resampling.NEAREST))

        return (torch.from_numpy(final_mask.astype(np.float32) / 255.0).unsqueeze(0),)

NODE_CLASS_MAPPINGS = {"GeminiSegmentationVertexNode": GeminiSegmentationVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiSegmentationVertexNode": "Gemini Segmentation (Vertex AI)"}