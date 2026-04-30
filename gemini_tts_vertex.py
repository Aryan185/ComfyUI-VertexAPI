import json
import torch
from google.genai import Client, types
from google.oauth2 import service_account

class GeminiTTSVertexNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
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
                "model": (["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],),
                "voice_id": (["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Achernar", "Laomedeia", "Rasalgethi", "Algenib", "Achird", "Pulcherrima", "Gacrux", "Schedar", "Alnilam", "Sulafat", "Sadaltager", "Sadachbia", "Vindemiatrix", "Zubenelgenubi"],),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"

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

        return Client(
            vertexai=True,
            project=project_id.strip(),
            location=location.strip(),
            credentials=credentials,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )

    def generate_speech(self, text, project_id, location, service_account, voice_id,
                        temperature, model, seed, system_prompt=""):

        if not text.strip():
            raise ValueError("Text input cannot be empty.")

        client = self.setup_client(service_account, project_id, location)

        final_prompt = f"{system_prompt.strip()}\n\n{text}" if system_prompt.strip() else text

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
                )
            )
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=final_prompt,
                config=config
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        try:
            audio_bytes = response.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but it contained no audio data.")

        waveform = torch.frombuffer(bytearray(audio_bytes), dtype=torch.int16)
        waveform = waveform.to(torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": 24000},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice_id', '')}-{kwargs.get('temperature', 1.0)}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('system_prompt', '')}"

NODE_CLASS_MAPPINGS = {"GeminiTTSVertexNode": GeminiTTSVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTSVertexNode": "Gemini TTS (Vertex AI)"}