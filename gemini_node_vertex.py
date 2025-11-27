import os
import io
import json
import tempfile
import numpy as np
import torch
from PIL import Image
from typing import Optional
from google import genai
from google.genai import types

class GeminiChatVertexNode:
    """ComfyUI Node for Gemini Chat via Vertex AI with optional image input"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
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
                    "gemini-3-pro-preview",
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash"
                ], {"default": "gemini-2.5-pro"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "thinking_budget": ("INT", {"default": -1, "min": -1, "max": 24576, "step": 1}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    
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
    
    def generate(self, prompt: str, project_id: str, location: str, service_account: str,
                 model: str, temperature: float, thinking: bool, seed: int,
                 system_instruction: Optional[str] = None, thinking_budget: int = -1, 
                 image: Optional[torch.Tensor] = None) -> tuple:   

        # Initialize Vertex AI client
        client = self.setup_client(service_account, project_id, location)
        parts = [types.Part.from_text(text=prompt)]
        
        # Handle image input
        if image is not None:
            img_array = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            if len(img_array.shape) == 4:
                img_array = img_array[0]
            if img_array.dtype in [np.float32, np.float64]:
                img_array = (img_array * 255).astype(np.uint8)
            
            buffered = io.BytesIO()
            Image.fromarray(img_array).save(buffered, format="PNG")
            parts.append(types.Part.from_bytes(mime_type="image/png", data=buffered.getvalue()))
        
        model_lower = model.lower()
        
        # Gemini 2.0 models don't support thinking at all
        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
            final_thinking_budget = None
        # Gemini Pro models (2.5-pro, 3-pro) cannot turn thinking off
        elif "pro" in model_lower and ("2.5" in model_lower or "gemini-3" in model_lower):
            print(f"{model} cannot have thinking turned off - thinking is always enabled")
            final_thinking_budget = thinking_budget if thinking_budget != 0 else -1
        # Flash models can toggle thinking on/off
        elif not thinking:
            final_thinking_budget = 0
        else:
            final_thinking_budget = thinking_budget
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_mime_type="text/plain"
        )
        
        if "gemini-2.0" not in model_lower:
            config.thinking_config = types.ThinkingConfig(thinking_budget=final_thinking_budget)
        
        if system_instruction and system_instruction.strip():
            config.system_instruction = [types.Part.from_text(text=system_instruction.strip())]
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config
        )
        
        return (response.text,)
            

# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeminiChatVertexNode": GeminiChatVertexNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiChatVertexNode": "Gemini Chat (Vertex AI)"
}