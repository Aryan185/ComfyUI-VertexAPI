import base64
import os
import io
import json
import tempfile
import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

class GeminiSegmentationVertexNode:
    """ComfyUI Node for Gemini Image Segmentation via Vertex AI"""
    
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
                "thinking": ("BOOLEAN", {"default": True}),
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
    
    def generate_segmentation(self, image: torch.Tensor, segment_prompt: str, project_id: str,
                            location: str, service_account: str, model: str, 
                            temperature: float, thinking: bool, seed: int,
                            thinking_budget: int = 0) -> tuple:
        
        # Initialize Vertex AI client
        client = self.setup_client(service_account, project_id, location)
        
        img_array = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        if len(img_array.shape) == 4:
            img_array = img_array[0]  # Remove batch dimension
        if img_array.dtype in [np.float32, np.float64]:
            img_array = (img_array * 255).astype(np.uint8)
        
        original_image = Image.fromarray(img_array).convert('RGB')
        original_width, original_height = original_image.size
        
        max_size = 1024
        scale = min(max_size / original_width, max_size / original_height)
        
        if scale < 1:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            processed_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            processed_image = original_image
        
        buffer = io.BytesIO()
        processed_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        base_prompt = f"Give the segmentation masks for {segment_prompt}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels. You are to only return only text output."
        
        parts = [
            types.Part.from_bytes(mime_type="image/png", data=image_data),
            types.Part.from_text(text=base_prompt)
        ]
        
        model_lower = model.lower()
        
        # Gemini 2.0 models don't support thinking at all
        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
            final_thinking_budget = None
        # Gemini Pro models (2.5-pro) cannot turn thinking off
        elif "pro" in model_lower and "2.5" in model_lower:
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
            response_mime_type="text/plain",
            response_modalities=["TEXT"]
        )
        
        if "gemini-2.0" not in model_lower:
            config.thinking_config = types.ThinkingConfig(thinking_budget=final_thinking_budget)
        
        # Generate content
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=config
            )
            
            response_text = response.text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            
            segments = json.loads(response_text)
            
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {str(e)}")
        
        # Create mask from segments
        proc_width, proc_height = processed_image.size
        mask_image = Image.new('L', (proc_width, proc_height), 0)
        
        # Sort segments by size (largest first)
        segments_with_size = []
        for segment in segments:
            box_2d = segment['box_2d']
            ymin, xmin, ymax, xmax = box_2d
            w = (xmax - xmin) / 1000
            h = (ymax - ymin) / 1000
            segments_with_size.append((segment, w * h))
        
        segments_with_size.sort(key=lambda x: x[1], reverse=True)
        
        # Process each segment
        for i, (segment, _) in enumerate(segments_with_size):
            try:
                box_2d = segment['box_2d']
                ymin, xmin, ymax, xmax = box_2d
                
                x = int(xmin / 1000 * proc_width)
                y = int(ymin / 1000 * proc_height)
                w = int((xmax - xmin) / 1000 * proc_width)
                h = int((ymax - ymin) / 1000 * proc_height)
                
                mask_data = segment['mask']
                
                if isinstance(mask_data, str):
                    if mask_data.startswith('data:image'):
                        mask_data = mask_data.split(',')[1]
                    mask_bytes = base64.b64decode(mask_data)
                    mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                else:
                    continue
                
                if mask_img.size != (w, h):
                    mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
                
                mask_array = list(mask_img.getdata())
                final_pixels = [255 if alpha > 128 else 0 for alpha in mask_array]
                segment_mask = Image.new('L', (w, h))
                segment_mask.putdata(final_pixels)
                
                if x + w <= proc_width and y + h <= proc_height and x >= 0 and y >= 0:
                    region = mask_image.crop((x, y, x + w, y + h))
                    region_pixels = list(region.getdata())
                    segment_pixels = list(segment_mask.getdata())
                    combined_pixels = [max(r, s) for r, s in zip(region_pixels, segment_pixels)]
                    combined_region = Image.new('L', (w, h))
                    combined_region.putdata(combined_pixels)
                    mask_image.paste(combined_region, (x, y))
                
            except Exception:
                continue
        
        if processed_image.size != original_image.size:
            mask_image = mask_image.resize(original_image.size, Image.Resampling.LANCZOS)
        
        # Convert PIL mask to ComfyUI mask format
        mask_array = np.array(mask_image, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add batch dimension
        
        return (mask_tensor,)

# Node mappings
NODE_CLASS_MAPPINGS = {"GeminiSegmentationVertexNode": GeminiSegmentationVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiSegmentationVertexNode": "Gemini Segmentation (Vertex AI)"}