"""
Question Generator Module for InteriorGS Dataset

This module combines object selection, camera sampling, and question construction
to generate spatial reasoning questions from InteriorGS scenes.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from .config import QuestionConfig
    from .object_selector import SceneObject
    from .camera_sampler import CameraPose
    from . import question_utils
    from . import question_templates
except ImportError:
    from config import QuestionConfig
    from object_selector import SceneObject
    from camera_sampler import CameraPose
    import question_utils
    import question_templates

random.seed(42)


class QuestionGenerator:
    """
    Generates questions from selected objects and camera poses.
    
    Supports three categories of questions:
    1. Single-object questions (size, distance to camera)
    2. Pair-object questions (size comparison, distance between objects)
    3. Multi-object questions (distance comparisons across multiple objects)
    """
    
    def __init__(self, config: QuestionConfig):
        self.config = config
    
    def generate_single_object_questions(self, obj: SceneObject, 
                                          camera_pose: CameraPose) -> List[Dict[str, Any]]:
        """
        Generate questions for a single object.
        
        Args:
            obj: The target object
            camera_pose: Camera pose for viewing the object
        
        Returns:
            List of question dictionaries
        """
        questions = []
        
        for q_type in question_templates.SINGLE_OBJECT_QUESTIONS:
            if q_type not in self.config.enabled_question_types:
                continue
            
            try:
                if q_type == 'object_size':
                    qa = question_utils.construct_object_size_qa(obj)
                elif q_type == 'object_distance_to_camera':
                    qa = question_utils.construct_object_distance_to_camera_qa(obj, camera_pose)
                else:
                    continue
                
                if qa:
                    qa['camera_pose'] = camera_pose.to_dict()
                    questions.append(qa)
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_pair_object_questions(self, obj1: SceneObject, obj2: SceneObject,
                                        camera_pose: CameraPose) -> List[Dict[str, Any]]:
        """
        Generate questions for a pair of objects.
        
        Args:
            obj1: First object
            obj2: Second object
            camera_pose: Camera pose for viewing the objects
        
        Returns:
            List of question dictionaries
        """
        questions = []
        
        for q_type in question_templates.PAIR_OBJECT_QUESTIONS:
            if q_type not in self.config.enabled_question_types:
                continue
            
            try:
                if q_type == 'object_size_comparison_relative':
                    for dimension in self.config.dimensions:
                        qa = question_utils.construct_object_size_comparison_relative_qa(
                            obj1, obj2, dimension
                        )
                        if qa:
                            qa['camera_pose'] = camera_pose.to_dict()
                            questions.append(qa)
                
                elif q_type == 'object_size_comparison_absolute':
                    for dimension in self.config.dimensions:
                        qa = question_utils.construct_object_size_comparison_absolute_qa(
                            obj1, obj2, dimension
                        )
                        if qa:
                            qa['camera_pose'] = camera_pose.to_dict()
                            questions.append(qa)
                
                elif q_type == 'object_pair_distance_center':
                    qa = question_utils.construct_object_pair_distance_center_qa(obj1, obj2)
                    if qa:
                        qa['camera_pose'] = camera_pose.to_dict()
                        questions.append(qa)
                
                elif q_type == 'object_pair_distance_center_w_size':
                    for dimension in self.config.dimensions:
                        qa = question_utils.construct_object_pair_distance_center_w_size_qa(
                            obj1, obj2, dimension
                        )
                        if qa:
                            qa['camera_pose'] = camera_pose.to_dict()
                            questions.append(qa)
                
                elif q_type == 'object_pair_distance_vector':
                    qa = question_utils.construct_object_pair_distance_vector_qa(
                        obj1, obj2, camera_pose
                    )
                    if qa:
                        questions.append(qa)
                
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_multi_object_questions(self, objects: List[SceneObject],
                                          camera_pose: CameraPose,
                                          max_questions_per_type: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multi-object comparison questions.
        
        Args:
            objects: List of all valid objects in the scene (need at least 3-4)
            camera_pose: Camera pose for viewing the objects
            max_questions_per_type: Maximum questions per question type
        
        Returns:
            List of question dictionaries
        """
        questions = []
        
        if len(objects) < 3:
            return questions
        
        for q_type in question_templates.MULTI_OBJECT_QUESTIONS:
            if q_type not in self.config.enabled_question_types:
                continue
            
            type_questions = []
            
            try:
                if q_type == 'object_comparison_absolute_distance':
                    # Need 4 objects: A, B, X, Y (can have overlap)
                    if len(objects) >= 3:
                        # Sample combinations
                        for _ in range(max_questions_per_type * 2):
                            if len(objects) >= 4:
                                sampled = random.sample(objects, 4)
                                obj_a, obj_b, obj_x, obj_y = sampled
                            else:
                                # Use 3 objects with one shared
                                sampled = random.sample(objects, 3)
                                obj_a = sampled[0]
                                obj_b = sampled[1]
                                obj_x = sampled[0]  # Shared object
                                obj_y = sampled[2]
                            
                            qa = question_utils.construct_object_comparison_absolute_distance_qa(
                                obj_a, obj_b, obj_x, obj_y
                            )
                            if qa:
                                qa['camera_pose'] = camera_pose.to_dict()
                                type_questions.append(qa)
                            
                            if len(type_questions) >= max_questions_per_type:
                                break
                
                elif q_type == 'object_comparison_relative_distance':
                    if len(objects) >= 3:
                        for _ in range(max_questions_per_type * 2):
                            if len(objects) >= 4:
                                sampled = random.sample(objects, 4)
                                obj_a, obj_b, obj_x, obj_y = sampled
                            else:
                                sampled = random.sample(objects, 3)
                                obj_a = sampled[0]
                                obj_b = sampled[1]
                                obj_x = sampled[0]
                                obj_y = sampled[2]
                            
                            qa = question_utils.construct_object_comparison_relative_distance_qa(
                                obj_a, obj_b, obj_x, obj_y
                            )
                            if qa:
                                qa['camera_pose'] = camera_pose.to_dict()
                                type_questions.append(qa)
                            
                            if len(type_questions) >= max_questions_per_type:
                                break
                
                questions.extend(type_questions[:max_questions_per_type])
                
            except Exception as e:
                print(f"Warning: Failed to generate {q_type} question: {e}")
        
        return questions
    
    def generate_all_questions(self, objects: List[SceneObject],
                                pairs: List[Tuple[SceneObject, SceneObject]],
                                camera_pose: CameraPose) -> List[Dict[str, Any]]:
        """
        Generate all types of questions for given objects and camera pose.
        
        Args:
            objects: List of single objects
            pairs: List of object pairs
            camera_pose: Camera pose
        
        Returns:
            List of all generated question dictionaries
        """
        all_questions = []
        
        # Single object questions
        for obj in objects:
            questions = self.generate_single_object_questions(obj, camera_pose)
            all_questions.extend(questions)
        
        # Pair object questions
        for obj1, obj2 in pairs:
            questions = self.generate_pair_object_questions(obj1, obj2, camera_pose)
            all_questions.extend(questions)
        
        # Multi-object questions
        if len(objects) >= 3:
            questions = self.generate_multi_object_questions(
                objects, camera_pose, self.config.max_questions_per_type
            )
            all_questions.extend(questions)
        
        return all_questions
    
    def get_question_statistics(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics about generated questions by type."""
        stats = {}
        for q in questions:
            q_type = q.get('question_type', 'unknown')
            stats[q_type] = stats.get(q_type, 0) + 1
        return stats
