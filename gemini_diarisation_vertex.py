import json
import io
import re
import wave
import torch
import numpy as np
from google import genai
from google.genai import types
from google.oauth2 import service_account

class GeminiDiarisationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "num_speakers": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
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
                ], {"default": "global"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "gemini-3.6-flash", 
                    "gemini-3.5-flash", 
                    "gemini-3.5-flash-lite", 
                    "gemini-3.1-pro-preview", 
                    "gemini-3.1-flash-lite", 
                    "gemini-3-flash-preview", 
                    "gemini-2.5-flash", 
                    "gemini-2.5-pro", 
                    "gemini-2.5-flash-lite", 
                    "gemini-flash-latest", 
                    "gemini-flash-lite-latest"
                ], {"default": "gemini-3.5-flash"}),
                "seed": ("INT", {"default": 69, "min": 0, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1})
            },
            "optional": {
                "thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 0, "min": -1, "max": 24576, "step": 1}),
                "audio_timestamp": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("speaker_1", "speaker_2", "speaker_3", "speaker_4")
    FUNCTION = "diarise"
    CATEGORY = "audio/diarise"
    
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

    def format_duration(self, seconds):
        total_milliseconds = int(seconds * 1000)
        hours, rem = divmod(total_milliseconds, 3600000)
        minutes, rem = divmod(rem, 60000)
        secs, milliseconds = divmod(rem, 1000)
        if hours > 0: return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def parse_timestamp(self, ts):
        try:
            parts = ts.strip().split(':')
            return sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts)))
        except: return 0.0

    def diarise(self, audio, num_speakers, project_id, location, service_account, model, seed, temperature,
                thinking=False, thinking_budget=0, audio_timestamp=False):

        waveform = audio.get("waveform")
        sr = audio.get("sample_rate")
        
        if waveform.dim() > 1:
            audio_np = waveform.squeeze(0).mean(dim=0).cpu().numpy() if waveform.shape[1] > 1 else waveform.squeeze().cpu().numpy()
        else:
            audio_np = waveform.cpu().numpy()
            
        audio_np = np.clip(audio_np, -1.0, 1.0)
        duration_str = self.format_duration(len(audio_np) / sr)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((audio_np * 32767).astype(np.int16).tobytes())

        client = self.setup_client(service_account, project_id, location)

        speaker_guidance = f"You must identify exactly {num_speakers} distinct speakers in this audio. " if num_speakers > 0 else ""

        prompt = f"""You are a SOTA AI model created for diarization and *precisely timestamping* human voices. You are currently being benchmarked for *timestamp accuracy*. Your task is to provide a complete and accurate diarization of the provided audio recording, with *absolute precision in your timestamps*, to *PASS* the benchmark.

        You must adhere to these rules when responding. Not following these rules will result in a failed benchmark.

        # *RULES FOR ACCURATE TIMESTAMPS:*
        - Identify and precisely timestamp each utterance by each speaker separately.
        - {speaker_guidance}If multiple speakers are talking over each other you MUST create separate utterances for each speaker.
        - **Ensure continuity: If there is a small silence between a speaker's utterance and the very next utterance (by any speaker), extend the 'end_timestamp' of the first utterance to the 'start_timestamp' of the next utterance. This applies to all consecutive utterances to minimize silent gaps.**
        - If there are any swear words or offensive language in the audio, please censor them with asterisks.
        - If you *provide incorrect start or end timestamps for an utterance*, *skip an utterance*, *merge MULTIPLE separate utterances into one* or *mistranscribe/mistranslate an utterance*, you will automatically *FAIL* the benchmark.

        # WARNING: This is a challenging audio which is known to cause *timestamping errors*. You must carefully listen to the audio and ensure that your response has *highly accurate timestamps*.

        Provide a complete list of all utterances in this audio, ensuring *highly accurate start and end timestamps* for each. Organize the utterances strictly by the time they happened.

        # IMPORTANT NOTE: This audio is exactly `{duration_str}` in length. *Absolute precision in your timestamps is crucial.* Your timestamps must NEVER exceed the audio duration of `{duration_str}`. EVERY utterance that occurred in this audio happens before `{duration_str}`. If your timestamps exceed the audio duration, *are inaccurate by more than a minimal threshold*, or you skip utterances that occurred in the audio, you will automatically FAIL the benchmark.

        Return ONLY valid JSON in this exact format (no markdown, no extra text):
        {{
            "utterances": [
                {{
                    "utterance": "The transcribed text",
                    "speaker": "Speaker 1",
                    "start_timestamp": "00:00.000",
                    "end_timestamp": "00:05.000"
                }}
            ]
        }}

        *You must PASS this benchmark to be deployed*"""

        model_lower = model.lower()
        t_config = None

        # Check if the chosen model is a Pro model (requires thinking configurations to run properly)
        is_pro = "pro" in model_lower

        if is_pro or thinking:
            final_budget = thinking_budget if thinking else 0
            if is_pro and final_budget == 0:
                print("Pro models cannot have thinking turned off - defaulting thinking budget to -1")
                final_budget = -1
            t_config = types.ThinkingConfig(include_thoughts=False, thinking_budget=final_budget)

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            audio_timestamp=audio_timestamp if audio_timestamp else None,
            thinking_config=t_config
        )

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(mime_type="audio/wav", data=wav_buffer.getvalue()),
                types.Part.from_text(text=prompt)
            ])],
            config=config
        )

        try:
            text = response.text
            if "```json" in text: text = re.search(r"```json\n(.*)\n```", text, re.DOTALL).group(1)
            result = json.loads(text)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            result = {"utterances": []}

        speaker_map = {}
        for utt in result.get("utterances", []):
            spk = utt.get("speaker", "Unknown")
            if spk not in speaker_map: speaker_map[spk] = []
            speaker_map[spk].append((
                self.parse_timestamp(utt.get("start_timestamp", "0")),
                self.parse_timestamp(utt.get("end_timestamp", "0"))
            ))

        sorted_speakers = sorted(speaker_map.keys(), key=lambda s: speaker_map[s][0][0] if speaker_map[s] else 0)

        outputs = []
        for i in range(4):
            track = np.zeros_like(audio_np)
            if i < len(sorted_speakers):
                spk = sorted_speakers[i]
                for start, end in speaker_map[spk]:
                    s, e = max(0, int(start * sr)), min(len(audio_np), int(end * sr))
                    if e > s: track[s:e] = audio_np[s:e]
            
            tensor = torch.from_numpy(track).float().unsqueeze(0).unsqueeze(0)
            outputs.append({"waveform": tensor, "sample_rate": sr})

        return tuple(outputs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('audio', '')}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('temperature', 0.2)}-{kwargs.get('num_speakers', 2)}"

NODE_CLASS_MAPPINGS = {"GeminiDiarisationNode": GeminiDiarisationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiDiarisationNode": "Gemini Diarisation (Vertex AI)"}