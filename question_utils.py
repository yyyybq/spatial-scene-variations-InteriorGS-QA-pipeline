"""
Question Utilities for InteriorGS Dataset

This module contains utility functions for constructing QA pairs from InteriorGS data,
adapted from sceneshift/question_generation/question_utils.py.
"""

import math
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R

try:
    from . import question_templates
    from .object_selector import SceneObject
    from .camera_sampler import CameraPose
except ImportError:
    import question_templates
    from object_selector import SceneObject
    from camera_sampler import CameraPose

random.seed(42)


def get_object_size(obj: SceneObject) -> Tuple[float, float, float]:
    """
    Get object size as (length, width, height).
    
    Length is the longer of the two horizontal dimensions.
    Width is the shorter of the two horizontal dimensions.
    Height is the vertical dimension.
    """
    return obj.get_obb_size()


def calculate_distance(obj1: SceneObject, obj2: SceneObject) -> float:
    """Calculate Euclidean distance between two object centers."""
    return float(np.linalg.norm(obj1.center - obj2.center))


def calculate_distance_to_camera(obj: SceneObject, camera_pose: CameraPose) -> float:
    """Calculate distance from object center to camera position."""
    return float(np.linalg.norm(obj.center - camera_pose.position))


def transform_vector_to_camera_space(vector: np.ndarray, camera_pose: CameraPose) -> np.ndarray:
    """
    Transform a world-space vector to camera local coordinate system.
    
    Camera coordinate system:
    - X-axis: Left-to-right (positive X is right)
    - Y-axis: Up-and-down (positive Y is up)
    - Z-axis: Forward-and-backward (positive Z is away from camera)
    """
    # Get camera rotation from yaw and pitch
    # yaw: rotation around Z (world up), pitch: rotation around X (right)
    r = R.from_euler('yxz', [camera_pose.yaw, -camera_pose.pitch, 0], degrees=True)
    
    # Apply inverse rotation to get local vector
    local_vector = r.inv().apply(vector)
    
    return local_vector


# ==============================================================================
# SINGLE OBJECT QUESTION CONSTRUCTORS
# ==============================================================================

def construct_object_size_qa(obj: SceneObject, dimension: str = 'length') -> Dict[str, Any]:
    """
    Constructs a question about a single dimension (length, width, or height) of an object.
    """
    obj_length, obj_width, obj_height = get_object_size(obj)
    
    if dimension == 'length':
        value = obj_length
    elif dimension == 'width':
        value = obj_width
    else:  # height
        value = obj_height
    
    rounded_value = round(value, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_TEMPLATE.format(object=obj.label, dimension=dimension),
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_value),
        "answer_type": "numerical",
        "question_type": "object_size",
        "question_id": f"object_size_{obj.id}_{dimension}",
        "primary_object": obj.id,
        "objects": [obj.to_dict()],
    }


def construct_object_distance_to_camera_qa(obj: SceneObject, camera_pose: CameraPose) -> Dict[str, Any]:
    """
    Constructs a question about the distance of an object from the camera.
    """
    distance = calculate_distance_to_camera(obj, camera_pose)
    rounded_distance = round(distance, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_DISTANCE_TO_CAMERA_TEMPLATE.format(object=obj.label),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_distance),
        "answer_type": "numerical",
        "question_type": "object_distance_to_camera",
        "question_id": f"object_distance_to_camera_{obj.id}",
        "primary_object": obj.id,
        "objects": [obj.to_dict()],
        "camera_pose": camera_pose.to_dict(),
    }


# ==============================================================================
# PAIR OBJECT QUESTION CONSTRUCTORS
# ==============================================================================

def construct_object_size_comparison_relative_qa(obj1: SceneObject, obj2: SceneObject, 
                                                   dimension: str) -> Optional[Dict[str, Any]]:
    """
    Constructs a question comparing the relative size of two objects along a specific dimension.
    """
    obj1_length, obj1_width, obj1_height = get_object_size(obj1)
    obj2_length, obj2_width, obj2_height = get_object_size(obj2)
    
    if dimension == "length":
        ratio = obj1_length / (obj2_length + 1e-9)
    elif dimension == "width":
        ratio = obj1_width / (obj2_width + 1e-9)
    else:  # height
        ratio = obj1_height / (obj2_height + 1e-9)
    
    rounded_ratio = round(ratio, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_COMPARISON_RELATIVE_TEMPLATE.format(
                dimension=dimension, object1=obj1.label, object2=obj2.label
            ),
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_ratio),
        "answer_type": "numerical",
        "question_type": "object_size_comparison_relative",
        "question_id": f"object_size_comparison_relative_{obj1.id}_{obj2.id}_{dimension}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
    }


def construct_object_size_comparison_absolute_qa(obj1: SceneObject, obj2: SceneObject,
                                                   dimension: str) -> Optional[Dict[str, Any]]:
    """
    Constructs a question comparing the absolute size of two objects along a specific dimension.
    """
    obj1_length, obj1_width, obj1_height = get_object_size(obj1)
    obj2_length, obj2_width, obj2_height = get_object_size(obj2)
    
    if dimension == "length":
        obj1_dim = obj1_length
        obj2_dim = obj2_length
    elif dimension == "width":
        obj1_dim = obj1_width
        obj2_dim = obj2_width
    else:  # height
        obj1_dim = obj1_height
        obj2_dim = obj2_height
    
    rounded_obj1_dim = round(obj1_dim, 2)
    rounded_obj2_dim = round(obj2_dim, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_SIZE_COMPARISON_ABSOLUTE_TEMPLATE.format(
                dimension=dimension, 
                object1=obj1.label, 
                obj2_dimension=rounded_obj2_dim, 
                object2=obj2.label
            ),
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_obj1_dim),
        "answer_type": "numerical",
        "question_type": "object_size_comparison_absolute",
        "question_id": f"object_size_comparison_absolute_{obj1.id}_{obj2.id}_{dimension}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
    }


def construct_object_pair_distance_center_qa(obj1: SceneObject, obj2: SceneObject) -> Dict[str, Any]:
    """
    Constructs a question about the distance between the centers of two objects.
    """
    distance = calculate_distance(obj1, obj2)
    rounded_distance = round(distance, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_CENTER_TEMPLATE.format(
                object1=obj1.label, object2=obj2.label
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_distance),
        "answer_type": "numerical",
        "question_type": "object_pair_distance_center",
        "question_id": f"object_pair_distance_center_{obj1.id}_{obj2.id}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
    }


def construct_object_pair_distance_center_w_size_qa(obj1: SceneObject, obj2: SceneObject,
                                                      dimension: str) -> Optional[Dict[str, Any]]:
    """
    Constructs a question about the distance between two objects, 
    relative to a specific dimension of the first object.
    """
    distance = calculate_distance(obj1, obj2)
    obj1_length, obj1_width, obj1_height = get_object_size(obj1)
    
    if dimension == "length":
        reference_dim = obj1_length
    elif dimension == "width":
        reference_dim = obj1_width
    else:  # height
        reference_dim = obj1_height
    
    if reference_dim < 0.01:
        return None  # Avoid issues with very small dimensions
    
    rounded_distance = round(distance, 2)
    rounded_ref_dim = round(reference_dim, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_CENTER_W_SIZE_TEMPLATE.format(
                object1=obj1.label,
                object2=obj2.label,
                dimension=dimension,
                obj1_dimension=rounded_ref_dim
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_distance),
        "answer_type": "numerical",
        "question_type": "object_pair_distance_center_w_size",
        "question_id": f"object_pair_distance_center_w_size_{obj1.id}_{obj2.id}_{dimension}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
    }


def construct_object_pair_distance_vector_qa(obj_a: SceneObject, obj_b: SceneObject,
                                               camera_pose: CameraPose) -> Dict[str, Any]:
    """
    Constructs a question asking for the vector from one object to another.
    """
    # Calculate the vector from object A to object B in world space
    world_vector = obj_b.center - obj_a.center
    
    # Transform to camera local space
    local_vector = transform_vector_to_camera_space(world_vector, camera_pose)
    
    # Round components
    rounded_x = round(local_vector[0], 2)
    rounded_y = round(local_vector[1], 2)
    rounded_z = round(local_vector[2], 2)
    
    answer_string = f"[{rounded_x}, {rounded_y}, {rounded_z}]"
    
    return {
        "question": " ".join([
            question_templates.OBJECT_PAIR_DISTANCE_VECTOR_TEMPLATE.format(
                objectA=obj_a.label, objectB=obj_b.label
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.VECTOR_POST_PROMPT,
            question_templates.POST_PROMPT_VECTOR
        ]),
        "answer": answer_string,
        "answer_type": "numerical",
        "question_type": "object_pair_distance_vector",
        "question_id": f"object_pair_distance_vector_{obj_a.id}_{obj_b.id}",
        "primary_object": obj_a.id,
        "objects": [obj_a.to_dict(), obj_b.to_dict()],
        "camera_pose": camera_pose.to_dict(),
    }


# ==============================================================================
# MULTI OBJECT QUESTION CONSTRUCTORS
# ==============================================================================

def construct_object_comparison_absolute_distance_qa(obj_a: SceneObject, obj_b: SceneObject,
                                                       obj_x: SceneObject, obj_y: SceneObject) -> Dict[str, Any]:
    """
    Constructs a question that provides the distance between two objects (X and Y) 
    and asks for the distance between two other objects (A and B).
    """
    dist_a_b = calculate_distance(obj_a, obj_b)
    dist_x_y = calculate_distance(obj_x, obj_y)
    
    rounded_dist_ab = round(dist_a_b, 2)
    rounded_dist_xy = round(dist_x_y, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_COMPARISON_ABSOLUTE_DISTANCE_TEMPLATE.format(
                objectA=obj_a.label,
                objectB=obj_b.label,
                distance=rounded_dist_xy,
                objectX=obj_x.label,
                objectY=obj_y.label
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_dist_ab),
        "answer_type": "numerical",
        "question_type": "object_comparison_absolute_distance",
        "question_id": f"object_comparison_absolute_distance_{obj_a.id}_{obj_b.id}_{obj_x.id}_{obj_y.id}",
        "primary_object": obj_a.id,
        "objects": [obj_a.to_dict(), obj_b.to_dict(), obj_x.to_dict(), obj_y.to_dict()],
    }


def construct_object_comparison_relative_distance_qa(obj_a: SceneObject, obj_b: SceneObject,
                                                       obj_x: SceneObject, obj_y: SceneObject) -> Dict[str, Any]:
    """
    Constructs a question asking for the relative distance ratio between two pairs of objects.
    """
    dist_a_b = calculate_distance(obj_a, obj_b)
    dist_x_y = calculate_distance(obj_x, obj_y)
    
    if dist_x_y < 0.01:
        return None  # Avoid division by zero
    
    distance_ratio = dist_a_b / dist_x_y
    rounded_ratio = round(distance_ratio, 2)
    
    return {
        "question": " ".join([
            question_templates.OBJECT_COMPARISON_RELATIVE_DISTANCE_TEMPLATE.format(
                objectA=obj_a.label,
                objectB=obj_b.label,
                objectX=obj_x.label,
                objectY=obj_y.label
            ),
            question_templates.DISTANCE_POST_PROMPT,
            question_templates.NA_POST_PROMPT,
            question_templates.POST_PROMPT_NA
        ]),
        "answer": str(rounded_ratio),
        "answer_type": "numerical",
        "question_type": "object_comparison_relative_distance",
        "question_id": f"object_comparison_relative_distance_{obj_a.id}_{obj_b.id}_{obj_x.id}_{obj_y.id}",
        "primary_object": obj_a.id,
        "objects": [obj_a.to_dict(), obj_b.to_dict(), obj_x.to_dict(), obj_y.to_dict()],
    }


# ==============================================================================
# QUESTION CONSTRUCTOR REGISTRY
# ==============================================================================

# ==============================================================================
# RELATIVE (YES/NO) QUESTION CONSTRUCTORS
# ==============================================================================

def _shuffle_yes_no(correct_yes: bool) -> Tuple[str, str, Dict[str, str]]:
    """
    Shuffle Yes/No into A/B options. Returns (correct_letter, choices_str, choices_dict).
    """
    options = ["Yes", "No"]
    random.shuffle(options)
    labels = ['A', 'B']
    correct_text = "Yes" if correct_yes else "No"
    correct_letter = labels[options.index(correct_text)]
    choices_str = "\n".join(f"({labels[i]}) {options[i]}" for i in range(2))
    choices_dict = {labels[i]: options[i] for i in range(2)}
    return correct_letter, choices_str, choices_dict


def construct_relative_size_qa(obj1: SceneObject, obj2: SceneObject) -> Dict[str, Any]:
    """
    Constructs an MC question: Is object1 larger than object2?
    Compares by volume (length * width * height).
    """
    l1, w1, h1 = get_object_size(obj1)
    l2, w2, h2 = get_object_size(obj2)
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    correct_yes = vol1 > vol2
    correct_letter, choices_str, choices_dict = _shuffle_yes_no(correct_yes)
    
    base_question = question_templates.RELATIVE_SIZE_TEMPLATE.format(
        object1=obj1.label, object2=obj2.label
    )
    
    return {
        "question": " ".join([
            base_question + "\n\n" + choices_str,
            question_templates.MCA_POST_PROMPT,
            question_templates.POST_PROMPT_MC
        ]),
        "answer": correct_letter,
        "answer_text": "Yes" if correct_yes else "No",
        "answer_type": "mc",
        "question_type": "relative_size",
        "question_id": f"relative_size_{obj1.id}_{obj2.id}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
        "choices": choices_dict,
    }


def construct_relative_distance_qa(obj1: SceneObject, obj2: SceneObject,
                                    obj3: SceneObject) -> Dict[str, Any]:
    """
    Constructs an MC question: Is dist(obj1, obj2) > dist(obj1, obj3)?
    """
    dist_12 = calculate_distance(obj1, obj2)
    dist_13 = calculate_distance(obj1, obj3)
    correct_yes = dist_12 > dist_13
    correct_letter, choices_str, choices_dict = _shuffle_yes_no(correct_yes)
    
    base_question = question_templates.RELATIVE_DISTANCE_TEMPLATE.format(
        object1=obj1.label, object2=obj2.label, object3=obj3.label
    )
    
    return {
        "question": " ".join([
            base_question + "\n\n" + choices_str,
            question_templates.MCA_POST_PROMPT,
            question_templates.POST_PROMPT_MC
        ]),
        "answer": correct_letter,
        "answer_text": "Yes" if correct_yes else "No",
        "answer_type": "mc",
        "question_type": "relative_distance",
        "question_id": f"relative_distance_{obj1.id}_{obj2.id}_{obj3.id}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict(), obj3.to_dict()],
        "choices": choices_dict,
    }


def construct_relative_distance_to_camera_qa(obj1: SceneObject, obj2: SceneObject,
                                              camera_pose: CameraPose) -> Dict[str, Any]:
    """
    Constructs an MC question: Is obj1 further from the camera than obj2?
    """
    dist1 = calculate_distance_to_camera(obj1, camera_pose)
    dist2 = calculate_distance_to_camera(obj2, camera_pose)
    correct_yes = dist1 > dist2
    correct_letter, choices_str, choices_dict = _shuffle_yes_no(correct_yes)
    
    base_question = question_templates.RELATIVE_DISTANCE_TO_CAMERA_TEMPLATE.format(
        object1=obj1.label, object2=obj2.label
    )
    
    return {
        "question": " ".join([
            base_question + "\n\n" + choices_str,
            question_templates.MCA_POST_PROMPT,
            question_templates.POST_PROMPT_MC
        ]),
        "answer": correct_letter,
        "answer_text": "Yes" if correct_yes else "No",
        "answer_type": "mc",
        "question_type": "relative_distance_to_camera",
        "question_id": f"relative_distance_to_camera_{obj1.id}_{obj2.id}",
        "primary_object": obj1.id,
        "objects": [obj1.to_dict(), obj2.to_dict()],
        "choices": choices_dict,
        "camera_pose": camera_pose.to_dict(),
    }


# ==============================================================================
# MULTIPLE CHOICE (MC) QUESTION CONSTRUCTORS
# ==============================================================================

def _generate_mc_distractors(correct_value: float, num_distractors: int = 3) -> List[float]:
    """
    Generate plausible distractor values for a multiple choice question.
    Distractors are within ±50-200% of the correct value, rounded to 2 decimal places.
    """
    distractors = []
    attempts = 0
    while len(distractors) < num_distractors and attempts < 50:
        attempts += 1
        # Random multiplier: 0.3x to 2.5x, avoiding 0.85-1.15 (too close)
        mult = random.choice([
            random.uniform(0.3, 0.7),
            random.uniform(1.3, 2.5),
        ])
        val = round(correct_value * mult, 2)
        if val <= 0:
            val = round(random.uniform(0.1, 2.0), 2)
        if val not in distractors and val != correct_value:
            distractors.append(val)
    # Fallback if not enough distractors
    while len(distractors) < num_distractors:
        val = round(correct_value + random.uniform(0.5, 3.0) * random.choice([-1, 1]), 2)
        if val <= 0:
            val = round(random.uniform(0.1, 2.0), 2)
        if val not in distractors and val != correct_value:
            distractors.append(val)
    return distractors


def _format_mc_question(base_question: str, correct_value: float,
                         question_id_base: str, primary_object: str,
                         objects_list: list, mc_source_type: str) -> Dict[str, Any]:
    """
    Wrap a numerical question into MC format with A/B/C/D choices.
    Only uses MCA_POST_PROMPT + POST_PROMPT_MC (no numerical or other conflicting prompts).
    """
    distractors = _generate_mc_distractors(correct_value, num_distractors=3)
    all_values = [correct_value] + distractors
    random.shuffle(all_values)

    labels = ['A', 'B', 'C', 'D']
    correct_letter = labels[all_values.index(correct_value)]

    choices_str = "\n".join(
        f"({labels[i]}) {all_values[i]}" for i in range(4)
    )

    full_question = " ".join([
        question_templates.MC_TEMPLATE.format(
            base_question=base_question,
            choices=choices_str
        ),
        question_templates.MCA_POST_PROMPT,
        question_templates.POST_PROMPT_MC,
    ])

    return {
        "question": full_question,
        "answer": correct_letter,
        "answer_value": correct_value,
        "answer_type": "mc",
        "question_type": "mc",
        "mc_source_type": mc_source_type,
        "question_id": f"mc_{question_id_base}",
        "primary_object": primary_object,
        "objects": objects_list,
        "choices": {labels[i]: all_values[i] for i in range(4)},
    }


def construct_mc_object_size_qa(obj: SceneObject, dimension: str = 'length') -> Optional[Dict[str, Any]]:
    """MC version of object_size: What is the {dimension} of {object}?"""
    obj_length, obj_width, obj_height = get_object_size(obj)
    value_map = {'length': obj_length, 'width': obj_width, 'height': obj_height}
    value = round(value_map[dimension], 2)

    base_q = question_templates.OBJECT_SIZE_TEMPLATE.format(object=obj.label, dimension=dimension)
    return _format_mc_question(
        base_question=base_q,
        correct_value=value,
        question_id_base=f"object_size_{obj.id}_{dimension}",
        primary_object=obj.id,
        objects_list=[obj.to_dict()],
        mc_source_type="object_size",
    )


def construct_mc_object_distance_to_camera_qa(obj: SceneObject, camera_pose: CameraPose) -> Optional[Dict[str, Any]]:
    """MC version of object_distance_to_camera."""
    distance = round(calculate_distance_to_camera(obj, camera_pose), 2)

    base_q = question_templates.OBJECT_DISTANCE_TO_CAMERA_TEMPLATE.format(object=obj.label)
    result = _format_mc_question(
        base_question=base_q,
        correct_value=distance,
        question_id_base=f"object_distance_to_camera_{obj.id}",
        primary_object=obj.id,
        objects_list=[obj.to_dict()],
        mc_source_type="object_distance_to_camera",
    )
    result["camera_pose"] = camera_pose.to_dict()
    return result


def construct_mc_object_pair_distance_center_qa(obj1: SceneObject, obj2: SceneObject) -> Optional[Dict[str, Any]]:
    """MC version of object_pair_distance_center."""
    distance = round(calculate_distance(obj1, obj2), 2)

    base_q = question_templates.OBJECT_PAIR_DISTANCE_CENTER_TEMPLATE.format(
        object1=obj1.label, object2=obj2.label
    )
    return _format_mc_question(
        base_question=base_q,
        correct_value=distance,
        question_id_base=f"object_pair_distance_center_{obj1.id}_{obj2.id}",
        primary_object=obj1.id,
        objects_list=[obj1.to_dict(), obj2.to_dict()],
        mc_source_type="object_pair_distance_center",
    )


# Maps question type to constructor function
QUESTION_CONSTRUCTORS = {
    'object_size': construct_object_size_qa,
    'object_distance_to_camera': construct_object_distance_to_camera_qa,
    'object_size_comparison_relative': construct_object_size_comparison_relative_qa,
    'object_size_comparison_absolute': construct_object_size_comparison_absolute_qa,
    'object_pair_distance_center': construct_object_pair_distance_center_qa,
    'object_pair_distance_center_w_size': construct_object_pair_distance_center_w_size_qa,
    'object_pair_distance_vector': construct_object_pair_distance_vector_qa,
    'object_comparison_absolute_distance': construct_object_comparison_absolute_distance_qa,
    'object_comparison_relative_distance': construct_object_comparison_relative_distance_qa,
    'relative_size': construct_relative_size_qa,
    'relative_distance': construct_relative_distance_qa,
    'relative_distance_to_camera': construct_relative_distance_to_camera_qa,
}
