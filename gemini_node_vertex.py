import os
import io
import json
import tempfile
import wave
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

class GeminiChatVertexNode:
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
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-3-pro-preview", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "thinking_budget": ("INT", {"default": -1, "min": -1, "max": 24576, "step": 1}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    
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
        
        return genai.Client(
            vertexai=True,
            project=project_id.strip(),
            location=location.strip(),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )
    
    def generate(self, prompt, project_id, location, service_account, model, temperature, thinking, seed,
                 system_instruction=None, thinking_budget=-1, image=None, audio=None):   

        client = self.setup_client(service_account, project_id, location)
        parts = [types.Part.from_text(text=prompt)]
        
        if image is not None:
            arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            parts.append(types.Part.from_bytes(mime_type="image/png", data=buf.getvalue()))
        
        if audio is not None:
            wf = audio.get("waveform") if isinstance(audio, dict) else audio[0]
            sr = audio.get("sample_rate", 44100) if isinstance(audio, dict) else audio[1]
            
            wf = wf.cpu().numpy() if isinstance(wf, torch.Tensor) else wf
            if wf.ndim > 1: wf = wf.mean(axis=0) if wf.shape[0] > 1 else wf.squeeze()
            wf_int16 = (np.clip(wf, -1, 1) * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(wf_int16.tobytes())
            parts.append(types.Part.from_bytes(mime_type="audio/wav", data=buf.getvalue()))
        
        model_lower = model.lower()
        t_config = None

        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
        else:
            final_budget = 0 
            
            if not thinking:
                if "gemini-2.5-pro" in model_lower or "gemini-3-pro-preview" in model_lower:
                    print("Pro models cannot have thinking turned off - defaulting thinking budget to -1")
                    final_budget = -1
            else:
                final_budget = thinking_budget
                if ("gemini-2.5-pro" in model_lower or "gemini-3-pro-preview" in model_lower) and final_budget == 0:
                    print("Pro models cannot have thinking turned off - defaulting thinking budget to -1")
                    final_budget = -1
            
            t_config = types.ThinkingConfig(thinking_budget=final_budget)

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            system_instruction=system_instruction.strip() if system_instruction else None,
            thinking_config=t_config
        )
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config
        )
        
        return (response.text,)

NODE_CLASS_MAPPINGS = {"GeminiChatVertexNode": GeminiChatVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiChatVertexNode": "Gemini Chat (Vertex AI)"}