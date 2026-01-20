"""
Question Generation Pipeline for InteriorGS Dataset

This module generates spatial reasoning questions from the InteriorGS dataset,
following the question categories from sceneshift/question_generation.
"""

from .config import PipelineConfig, ObjectSelectionConfig, CameraSamplingConfig
from .object_selector import ObjectSelector, SceneObject
from .camera_sampler import CameraSampler, CameraPose
from .question_generator import QuestionGenerator
from .pipeline import InteriorGSQuestionPipeline

__all__ = [
    'PipelineConfig',
    'ObjectSelectionConfig', 
    'CameraSamplingConfig',
    'ObjectSelector',
    'SceneObject',
    'CameraSampler',
    'CameraPose',
    'QuestionGenerator',
    'InteriorGSQuestionPipeline',
]
