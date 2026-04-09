"""
SceneShift-Bench Question Generation Pipeline (InteriorGS)

Generates spatial reasoning VQA from InteriorGS 3D indoor scenes.
50 curated scenes, 13 question types, 4 camera trajectory modes.
"""

from .config import PipelineConfig, ObjectSelectionConfig, CameraSamplingConfig
from .object_selector import ObjectSelector, SceneObject
from .camera_sampler import CameraSampler, CameraPose
from .question_generator import QuestionGenerator
from .scenes import DEFAULT_SCENES, QUESTION_TYPES, PATTERNS

__all__ = [
    'PipelineConfig',
    'ObjectSelectionConfig',
    'CameraSamplingConfig',
    'ObjectSelector',
    'SceneObject',
    'CameraSampler',
    'CameraPose',
    'QuestionGenerator',
    'DEFAULT_SCENES',
    'QUESTION_TYPES',
    'PATTERNS',
]
