from . import (
    gemini_node_vertex,
    nano_banana_vertex,
    gemini_tts_vertex,
    veo_vertex,
    gemini_diarisation_vertex,
    imagen_edit_vertex,
    imagen_vertex,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

modules = [
    gemini_node_vertex,
    nano_banana_vertex,
    gemini_tts_vertex,
    veo_vertex,
    gemini_diarisation_vertex,
    imagen_edit_vertex,
    imagen_vertex,
]

for module in modules:
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]