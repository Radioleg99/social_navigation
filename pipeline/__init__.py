"""
Pipeline: bridges HumanSSG scene-graph output → MoLMoSpaces social navigation.

Stage 1 (static scene):
    scene_graph.json  →  SceneDescription  →  layout JSON  →  MPPI nav in MoLMoSpaces

Stage 2 (dynamic scene, future):
    live agent states  →  LLM  →  updated SceneDescription  →  re-plan
"""
from .scene_bridge import scene_graph_to_scene_description
from .scene_builder import scene_description_to_layout_json, patch_layout_json

__all__ = [
    "scene_graph_to_scene_description",
    "scene_description_to_layout_json",
    "patch_layout_json",
]
