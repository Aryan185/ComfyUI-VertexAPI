import os
import io
import json
import tempfile
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

class NanoBananaVertexNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def setup_client(self, service_account_json, project_id, location):
        if not service_account_json.strip():
            raise ValueError("Service account JSON content is required.")
        if not project_id.strip():
            raise ValueError("Project ID is required.")
        
        try:
            json.loads(service_account_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write(service_account_json.strip())
        temp_file.close() 
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        
        return genai.Client(vertexai=True, project=project_id.strip(), location=location.strip())

    def _convert_tensor_to_bytes(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG')
        return buf.getvalue()

    def generate(self, project_id, location, service_account, model, aspect_ratio, resolution, temperature, top_p, seed,
                 prompt="", system_instruction="", **kwargs):
        
        client = self.setup_client(service_account, project_id, location)
        
        parts = []
        input_images = [kwargs.get(f"image_{i}") for i in range(1, 6)]
        for img in input_images:
            if img is not None:
                img_bytes = self._convert_tensor_to_bytes(img)
                parts.append(types.Part.from_bytes(mime_type="image/png", data=img_bytes))
        
        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))
            
        if not parts:
            raise ValueError("At least one image or prompt must be provided.")

        img_config_params = {"aspect_ratio": aspect_ratio}
        if "gemini-3-pro" in model:
            img_config_params["image_size"] = resolution
            
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(**img_config_params),
            system_instruction=system_instruction.strip() if system_instruction.strip() else None
        )
        
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")
        
        try:
            img_data = response.candidates[0].content.parts[0].inline_data.data
            result_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            result_tensor = torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0)
            return (result_tensor,)
            
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but no valid image data was found.")

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

NODE_CLASS_MAPPINGS = {"NanoBananaVertexNode": NanoBananaVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaVertexNode": "Nano Banana (Vertex AI)"}