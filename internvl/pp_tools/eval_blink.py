"""
Module for generating perception programs from BLINK tasks.
"""

import os
import json
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from pathlib import Path
import pp_tools as pp
from pp_tools import coords_to_perceptionprogram, coords_to_perceptionprogram_multi_image
import random


class PointAnnotationPromptGenerator:
    def __init__(self, point_annotations_path, key="idx"):
        self.point_annotations_path = point_annotations_path
        self.key = key

    def _load_point_annotations(self):
        """Load point annotations from JSONL file into a dictionary indexed by idx."""
        self.point_annotations = {}
        with open(self.point_annotations_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.point_annotations[entry[self.key]] = entry

    def _generate_point_annotations_perception_program(self, idx):
        # Get annotation entry
        if idx not in self.point_annotations:
            raise KeyError(f"No point annotations found for idx: {idx}")

        entry = self.point_annotations[idx]
        # Build point coordinates list from annotation coords
        coords = []
        for coord_data in entry["coords"]:
            coords.append({
                "x": int(round(coord_data["x"])),
                "y": int(round(coord_data["y"])),
                "label": coord_data["label"]
            })
        point_pp_str = coords_to_perceptionprogram(coords)
        return point_pp_str


class MultiImagePointAnnotationPromptGenerator(PointAnnotationPromptGenerator):
    """
    Extended version of PointAnnotationPromptGenerator that handles multi-image annotations.

    Merges multiple JSONL entries with the same idx but different image_field values,
    then generates perception programs using coords_to_perceptionprogram_multi_image.

    Args:
        point_annotations_path: Path to the JSONL file with point annotations
        key: Key to use for indexing entries (default: "idx")
        task: Task name to determine image labeling (default: "visual_correspondence")
    """

    # Define image name mappings for different tasks
    IMAGE_NAMES_BY_TASK = {
        "visual_correspondence": ("image 1 (reference)", "image 2 (target)"),
    }

    # Default image names if task not found
    DEFAULT_IMAGE_NAMES = ("image 1", "image 2", "image 3", "image 4")

    def __init__(self, point_annotations_path, task, key="idx"):
        super().__init__(point_annotations_path, key)
        self.task = task

        # Get image names for this task
        self.image_names = self.IMAGE_NAMES_BY_TASK.get(
            task,
            self.DEFAULT_IMAGE_NAMES
        )

    def _load_point_annotations(self):
        """
        Load point annotations from JSONL file, merging multiple image_field entries per idx.

        Each idx may have multiple lines in the JSONL (one per image_field).
        This method combines them into a single entry with coords organized by image_field.
        """
        self.point_annotations = {}
        with open(self.point_annotations_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                idx = entry[self.key]

                # Initialize if this idx hasn't been seen yet
                if idx not in self.point_annotations:
                    self.point_annotations[idx] = {
                        "idx": idx,
                        "coords": {},  # Dict mapping image_field -> coords list
                        "question": entry.get("question"),
                        "sub_task": entry.get("sub_task"),
                        "answer": entry.get("answer")
                    }

                # Add coords for this image_field
                image_field = entry["image_field"]
                self.point_annotations[idx]["coords"][image_field] = entry["coords"]

    def _get_image_name(self, image_field):
        """
        Get the human-readable image name for a given image_field.

        Args:
            image_field: Field name like "image_1", "image_2", etc.

        Returns:
            Human-readable image name from the task-specific tuple
        """
        # Extract image number from field name (e.g., "image_1" -> 0, "image_2" -> 1)
        try:
            image_num = int(image_field.split('_')[-1]) - 1  # Convert to 0-indexed

            # Get name from tuple, or use default format if index out of range
            if 0 <= image_num < len(self.image_names):
                return self.image_names[image_num]
            else:
                # Fallback for images beyond the tuple length
                return f"image {image_num + 1}"
        except (ValueError, IndexError):
            # Fallback if parsing fails
            return image_field

    def _generate_point_annotations_perception_program(self, idx):
        """
        Generate perception program for multi-image point annotations.

        Combines coordinates from all images, adds the 'image' field to each coord,
        and generates the perception program using coords_to_perceptionprogram_multi_image.
        """
        if idx not in self.point_annotations:
            raise KeyError(f"No point annotations found for idx: {idx}")

        entry = self.point_annotations[idx]
        coords_dict = entry["coords"]

        # Combine all coords from different images
        all_coords = []

        # Process in sorted order for consistency (image_1, image_2, etc.)
        for image_field in sorted(coords_dict.keys()):
            coords_list = coords_dict[image_field]

            # Get task-specific image name
            image_name = self._get_image_name(image_field)

            # Add coords with image field
            for coord_data in coords_list:
                all_coords.append({
                    "x": int(round(coord_data["x"])),
                    "y": int(round(coord_data["y"])),
                    "label": coord_data["label"],
                    "image": image_name
                })

        # Use multi-image perception program function
        point_pp_str = pp.coords_to_perceptionprogram_multi_image(all_coords)
        return point_pp_str


class SingleMainPPGenerator(metaclass=ABCMeta):
    @abstractmethod
    def _generate_perception_programs(self, idx):
        """
        Generate both main modality and point perception programs for a given image.

        Args:
            idx: Index to look up in point annotations

        Returns:
            tuple: (main_pp_str, point_pp_str) - both perception program strings

        Raises:
            FileNotFoundError: If modality image doesn't exist
            KeyError: If idx not found in point annotations
            ValueError: If modality image fails to load
        """
        pass

    def generate_prompt(self, question, idx):
        """
        Generate the full prompt with question and perception programs.

        Args:
            question: The question text to prepend to perception programs
            idx: Index to look up in point annotations (e.g., "val_Relative_Reflectance_1")

        Returns:
            str: Complete prompt with question and perception programs
        """
        main_pp_str, point_pp_str = self._generate_perception_programs(idx)
        return f"{question}\n{point_pp_str}\n{main_pp_str}"

class RelativeDepthPromptGenerator(PointAnnotationPromptGenerator, SingleMainPPGenerator):
    def __init__(self, field_root, point_annotations_path, grid=(10, 10), tau=0.15, relation_cap=0, **kwargs):
        super().__init__(point_annotations_path, **kwargs)
        self.root = field_root
        self.grid = grid
        self.tau = tau
        self.relation_cap = relation_cap
        self._load_point_annotations()
        
    def _generate_perception_programs(self, idx):
        # Generate point perception program
        point_pp_str = self._generate_point_annotations_perception_program(idx)

        # Derive depth image path from idx
        depth_filename = Path(idx).with_suffix('.png').name
        depth_image_path = os.path.join(self.root, depth_filename)

        # Validate depth image exists
        if not os.path.exists(depth_image_path):
            raise FileNotFoundError(f"depth image not found: {depth_image_path}")

        # Load depth map in grayscale
        depth_img = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            raise ValueError(f"Failed to load depth image: {depth_image_path}")

        # Normalize depth to [0, 1]
        depth_norm = depth_img.astype(np.float32) / 255.0

        # Generate depth perception program
        depth_pp_str, prog_json = pp.emit_perception_program(
            modality="depth",
            field=depth_norm,
            seg=None,
            class_names=None,
            grid=self.grid,
            add_relations=True,
            tau=self.tau,
            relation_cap=self.relation_cap
        )

        return depth_pp_str, point_pp_str


class VisualCorrespondencePromptGenerator(MultiImagePointAnnotationPromptGenerator, SingleMainPPGenerator):
    """
    Generator for visual correspondence perception program prompts.
    
    Handles multi-image point annotations and generates correspondence perception programs
    from precomputed keypoint matches.
    
    Attributes:
        correspondences_path: Path to JSONL file containing precomputed correspondences
        point_annotations_path: Path to JSONL file containing point annotations
        task: Task name for image labeling (default: "visual_correspondence")
    """
    
    def __init__(self, correspondences_path, point_annotations_path, task="visual_correspondence", key="idx", shuffle_randomly=False):
        """
        Initialize the prompt generator.
        
        Args:
            correspondences_path: Path to JSONL file containing correspondence data
            point_annotations_path: Path to JSONL file containing point annotations
            task: Task name to determine image labeling
            key: Key to use for indexing entries (default: "idx")
        """
        super().__init__(point_annotations_path, task, key)
        self.correspondences_path = correspondences_path
        self.shuffle_randomly = shuffle_randomly
        self._load_point_annotations()
        self._load_correspondences()

    def _load_point_annotations(self):
        """
        Load point annotations from JSONL file, merging multiple image_field entries per idx.
        If shuffle_randomly is True, shuffle (x, y) coordinates within each image field separately.
        """
        # Call parent method to load annotations normally
        super()._load_point_annotations()
        
        # If shuffling is enabled, shuffle coordinates within each image field
        if self.shuffle_randomly:
            for idx in self.point_annotations:
                # Process each image field separately
                for image_field in self.point_annotations[idx]["coords"]:
                    coords_list = self.point_annotations[idx]["coords"][image_field]
                    
                    # Collect all (x, y) coordinate pairs from this image field
                    xy_pairs = [(coord["x"], coord["y"]) for coord in coords_list]
                    
                    # Shuffle the coordinate pairs
                    random.shuffle(xy_pairs)
                    
                    # Reassign shuffled coordinates back (keeping labels in place)
                    for i, coord in enumerate(coords_list):
                        coord["x"] = xy_pairs[i][0]
                        coord["y"] = xy_pairs[i][1]
    
    def _load_correspondences(self):
        """Load correspondences from JSONL file into a dictionary indexed by idx."""
        self.correspondences = {}
        with open(self.correspondences_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if self.shuffle_randomly:
                    random.shuffle(entry["correspondences"]["kpts0"])
                    random.shuffle(entry["correspondences"]["kpts1"])
                self.correspondences[entry[self.key]] = entry["correspondences"]
    
    def _generate_perception_programs(self, idx):
        """
        Generate both correspondence and point perception programs for a given image pair.
        
        Args:
            idx: Index to look up in correspondences and point annotations
        
        Returns:
            tuple: (correspondence_pp_str, point_pp_str) - both perception program strings
        
        Raises:
            KeyError: If idx not found in correspondences or point annotations
        """
        # Generate multi-image point perception program
        point_pp_str = self._generate_point_annotations_perception_program(idx)
        
        # Get correspondence data for this idx
        if idx not in self.correspondences:
            raise KeyError(f"No correspondences found for idx: {idx}")
        
        correspondence_vals = self.correspondences[idx]
        
        # Generate correspondence perception program
        correspondence_pp_str, _ = pp.emit_perception_program(
            modality="correspondence",
            field=correspondence_vals,
            seg=None,
            class_names=None,
            grid=(10, 10),  # ignored for correspondence
            add_relations=False,  # not needed for correspondence
            tau=0.08,  # ignored for correspondence
            relation_cap=200
        )
        
        return correspondence_pp_str, point_pp_str