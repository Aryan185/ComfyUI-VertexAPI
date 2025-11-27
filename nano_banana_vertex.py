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
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
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
        """Setup Vertex AI client with service account JSON content"""
        if not service_account_json.strip():
            raise ValueError("Service account JSON content is required.")
        
        if not project_id.strip():
            raise ValueError("Project ID is required.")
        
        # Validate and write JSON content to temporary file
        try:
            json.loads(service_account_json)  # Validate JSON format
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")
        
        # Create temporary file with JSON content
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write(service_account_json.strip())
        temp_file.close()
        
        # Set credentials path
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        
        return genai.Client(vertexai=True, project=project_id.strip(), location=location.strip())
    
    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def pil_to_tensor(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)
        return tensor.unsqueeze(0)
    
    def generate(self, project_id, location, service_account, model, aspect_ratio, 
                 resolution, temperature, top_p, seed,
                 prompt="", system_instruction="", 
                 image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        
        # Initialize Vertex AI client
        client = self.setup_client(service_account, project_id, location)
        
        # Build parts list
        parts = []
        
        # Add images
        for img_tensor in [image_1, image_2, image_3, image_4, image_5]:
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                parts.append(types.Part.from_bytes(
                    mime_type="image/png",
                    data=buffer.getvalue()
                ))
        
        # Add prompt if provided
        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))
        
        if not parts:
            raise ValueError("At least one image or prompt must be provided.")
        
        contents = [types.Content(role="user", parts=parts)]
        
        config_dict = {
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(aspect_ratio=aspect_ratio),
        }

        if "gemini-3-pro" in model:
            config_dict["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio, image_size=resolution)
        else:
            print("gemini-2.5-flash-image does not support resolution parameter, using default resolution")

        config = types.GenerateContentConfig(**config_dict)
        
        if system_instruction.strip():
            config.system_instruction = [types.Part.from_text(text=system_instruction)]
        
        # Generate
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        
        result_image = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    result_image = Image.open(io.BytesIO(part.inline_data.data))
                    break
        
        if result_image is None:
            raise ValueError("No image generated by the API.")
        
        return (self.pil_to_tensor(result_image),)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('prompt', '')}-{kwargs.get('temperature', 0.5)}-{kwargs.get('top_p', 0.85)}-{kwargs.get('seed', 69)}-{kwargs.get('aspect_ratio', '1:1')}-{kwargs.get('model', 'gemini-3-pro-image-preview')}-{kwargs.get('image_1')}-{kwargs.get('image_2')}-{kwargs.get('image_3')}-{kwargs.get('image_4')}-{kwargs.get('image_5')}"

NODE_CLASS_MAPPINGS = {"NanoBananaVertexNode": NanoBananaVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaVertexNode": "Nano Banana (Vertex AI)"}