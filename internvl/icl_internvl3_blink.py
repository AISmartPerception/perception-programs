import textwrap

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from PIL import Image
import random
import json
import os
import time
import re
import argparse
from tqdm import tqdm
import pp_tools as pp
from pp_tools.eval_blink import VisualCorrespondencePromptGenerator
from pp_tools.utils import coords_to_perceptionprogram_multi_image, coords_to_perceptionprogram
from ml_utils.utils import batch_chat_multi_image
from prompts import PromptLoader

# Configuration defaults
SEED = 42
SUBSAMPLE_SIZE = None  # None means use full dataset
BATCH_SIZE = 8

# Generation configuration
MAX_NEW_TOKENS = 4600
FORCE_THINK_AFTER=4096

GENERATION_CONFIG = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
# GENERATION_CONFIG = dict(max_new_tokens=MAX_NEW_TOKENS,
#                          do_sample=True,
#                          temperature=0.6,
#                          num_beams=1,
#                          top_p=0.8,
#                          top_k=40
#                          )

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Problem-specific configurations
PROBLEM_CONFIGS = {
    'multi_view_reasoning': {
        'dataset_name': 'Multi-view_Reasoning',
        'dataset_key': 'Multi-view_Reasoning',
        'answer_choices': ['A', 'B'],  # Camera motion: left or right
    },
    'visual_correspondence': {
        'dataset_name': 'Visual_Correspondence',
        'dataset_key': 'Visual_Correspondence',
        'answer_choices': ['A', 'B', 'C', 'D', 'E'],  # Point matching
    },
    'hardblink3pointscenter': {
        'dataset_name': 'Relative_Depth_3points',
        'dataset_key': 'hardblink3pointscenter',
        'answer_choices': ['A', 'B', 'C'],  # Depth comparison with 3 points
    },
    'hardblink4pointscenter': {
        'dataset_name': 'Relative_Depth_4points',
        'dataset_key': 'hardblink4pointscenter',
        'answer_choices': ['A', 'B', 'C', 'D'],  # Depth comparison with 4 points
    },
    'hardblink5pointscenter': {
        'dataset_name': 'Relative_Depth_5points',
        'dataset_key': 'hardblink5pointscenter',
        'answer_choices': ['A', 'B', 'C', 'D', 'E'],  # Depth comparison with 5 points
    },
    'jigsaw': {
        'dataset_name': 'Jigsaw',
        'dataset_key': 'Jigsaw',
        'answer_choices': ['A', 'B'],  # Which candidate piece fits
    },
    'semantic_correspondence': {
        'dataset_name': 'Semantic_Correspondence',
        'dataset_key': 'Semantic_Correspondence',
        'answer_choices': ['A', 'B', 'C', 'D'],  # Which target point matches source
    },
    'object_localization': {
        'dataset_name': 'Object_Localization',
        'dataset_key': 'Object_Localization',
        'answer_choices': ['A', 'B'],  # Which bounding box is better
    }
}

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def setup_model(model_path, mode='pp', problem='multi_view_reasoning'):
    """
    Setup InternVL3 model with appropriate system message based on mode and problem.
    
    Args:
        model_path: Path to the model
        mode: 'pp' for perception programs, 'cot' for chain-of-thought, or 'tool' for visualization
        problem: Problem type ('multi_view_reasoning' or 'visual_correspondence')
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Load prompts using the PromptLoader
    prompt_loader = PromptLoader(problem=problem)
    system_prompt = prompt_loader.get_system_prompt(mode)
    
    # For pp and tool modes, append the ICL example to the system prompt
    if mode in ['pp', 'tool']:
        icl_example = prompt_loader.get_icl_example(mode)
        model.system_message = f"{system_prompt}\n\n{icl_example}"
    else:
        model.system_message = system_prompt
    
    return model, tokenizer

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_custom_jsonl_dataset(jsonl_path, image_dir):
    """
    Load a custom dataset from JSONL file for hardblink problems.
    
    Args:
        jsonl_path: Path to the JSONL file
        image_dir: Directory containing the images
        
    Returns:
        List of samples with 'idx', 'question', 'answer', 'prompt', 'image_1', 'image_2' keys
    """
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            img_path = os.path.join(image_dir, data['image'])
            img = Image.open(img_path).convert('RGB')
            
            # For relative depth, we only have one image per sample
            # We'll use the same image for both image_1 and image_2 to maintain compatibility
            sample = {
                'idx': data['image'].replace('.png', ''),
                'question': data['question'],
                'answer': f"({data['answer']})" if len(data["answer"]) == 1 else data["answer"],
                'prompt': '',  # No separate prompt prefix for these tasks
                'image_1': img,
                'image_2': img,  # Same image for single-image tasks
            }
            samples.append(sample)
    
    return samples

def call_internvl3_batch(model, tokenizer, questions, pp_prompts=None,
                         tool_imgs=None, tool_template=None, generation_config=None, 
                         problem='multi_view_reasoning', think_closing_tokens=FORCE_THINK_AFTER):
    """Call InternVL3 model for batch of samples using custom batch_chat"""
    if generation_config is None:
        generation_config = GENERATION_CONFIG
    
    # Prepare batch of prompts and pixel values
    all_pixel_values = []
    num_patches_lists = []
    batch_prompts = []
    
    for idx, question in enumerate(questions):
        img1 = question["image_1"]
        img2 = question.get('image_2')
        img3 = question.get('image_3')  # For jigsaw
        
        # For hardblink problems with PP, strip out perception program blocks
        if problem.startswith('hardblink') and pp_prompts is not None:
            # In PP mode, question field has PP embedded in <perceptionprogram> tags
            # Strip out all content between <perceptionprogram> and </perceptionprogram>
            question_text = re.sub(r'<perceptionprogram>.*?</perceptionprogram>', '', 
                                  question["question"], flags=re.DOTALL).strip()
        else:
            question_text = question["question"]

        question_ = question['prompt'] + "\n" + question_text if question['prompt'] else question_text
        
        # Load and preprocess images separately to track patches
        transform = build_transform(input_size=448)
        
        # Process first image
        images1 = dynamic_preprocess(img1, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values1 = torch.stack([transform(img) for img in images1])
        
        # Process second image (if present)
        pixel_values2 = None
        if img2 is not None:
            images2 = dynamic_preprocess(img2, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values2 = torch.stack([transform(img) for img in images2])
        
        # Process third image (for jigsaw)
        pixel_values3 = None
        if img3 is not None:
            images3 = dynamic_preprocess(img3, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values3 = torch.stack([transform(img) for img in images3])
        
        # Construct the prompt based on mode
        if tool_imgs is not None:
            # Tool mode - use visualization
            if problem == 'jigsaw':
                # Jigsaw needs TWO stitched images (one for each candidate)
                stitched_img1 = tool_imgs[idx][0]
                stitched_img2 = tool_imgs[idx][1]
                images_stitched1 = dynamic_preprocess(stitched_img1, image_size=448, use_thumbnail=True, max_num=12)
                images_stitched2 = dynamic_preprocess(stitched_img2, image_size=448, use_thumbnail=True, max_num=12)
                pixel_values_stitched1 = torch.stack([transform(img) for img in images_stitched1])
                pixel_values_stitched2 = torch.stack([transform(img) for img in images_stitched2])
                
                # For jigsaw: 3 original images + 2 stitched images
                all_pixel_values.append(pixel_values1)
                all_pixel_values.append(pixel_values2)
                all_pixel_values.append(pixel_values3)
                all_pixel_values.append(pixel_values_stitched1)
                all_pixel_values.append(pixel_values_stitched2)
                num_patches_lists.append([
                    pixel_values1.size(0), 
                    pixel_values2.size(0), 
                    pixel_values3.size(0),
                    pixel_values_stitched1.size(0),
                    pixel_values_stitched2.size(0)
                ])
                
                # Use template from config if provided
                if tool_template:
                    prompt = f"{tool_template}{question_}"
                else:
                    prompt = f"<image>\n<image>\n<image>\nStitched with A:\n<image>\nStitched with B:\n<image>\n{question_}"
            else:
                # For non-jigsaw problems, tool_imgs is a single image per sample
                images_tool = dynamic_preprocess(tool_imgs[idx], image_size=448, use_thumbnail=True, max_num=12)
                pixel_values_tool = torch.stack([transform(img) for img in images_tool])
                
                if problem.startswith('hardblink'):
                    # For hardblink, only use one image + depth map
                    all_pixel_values.append(pixel_values1)
                    all_pixel_values.append(pixel_values_tool)
                    num_patches_lists.append([pixel_values1.size(0), pixel_values_tool.size(0)])
                    
                    # Use template from config if provided, otherwise use default
                    if tool_template:
                        prompt = f"{tool_template}{question_}"
                    else:
                        prompt = f"Target image\n<image>\nDepth map\n<image>\n{question_}"
                elif problem == 'object_localization':
                    # For object_localization, use original image + visualization
                    all_pixel_values.append(pixel_values1)
                    all_pixel_values.append(pixel_values_tool)
                    num_patches_lists.append([pixel_values1.size(0), pixel_values_tool.size(0)])
                    
                    # Use template from config if provided, otherwise use default
                    if tool_template:
                        prompt = f"{tool_template}{question_}"
                    else:
                        prompt = f"Target image\n<image>\nDetections\n<image>\n{question_}"
                else:
                    # For other problems, use two images + tool visualization
                    all_pixel_values.append(pixel_values1)
                    all_pixel_values.append(pixel_values2)
                    all_pixel_values.append(pixel_values_tool)
                    num_patches_lists.append([pixel_values1.size(0), pixel_values2.size(0), pixel_values_tool.size(0)])
                    
                    # Use template from config if provided
                    if tool_template:
                        prompt = f"{tool_template}\n{question_}"
                    else:
                        # Fallback to default templates if not in config
                        if problem == 'multi_view_reasoning':
                            prompt = f"<image>\n<image>\n\nOptical flow:\n<image>\n\n{question_}"
                        else:  # visual_correspondence
                            prompt = f"<image>\n<image>\n\nVisual Correspondence Map:\n<image>\n\n{question_}"
        elif pp_prompts is not None:
            # PP mode
            if problem.startswith('hardblink'):
                # For hardblink, only use one image
                all_pixel_values.append(pixel_values1)
                num_patches_lists.append([pixel_values1.size(0)])
                prompt = f"{question_}\n{pp_prompts[idx]}\n<image>"
            elif problem == 'jigsaw':
                # For jigsaw, use three images
                all_pixel_values.append(pixel_values1)
                all_pixel_values.append(pixel_values2)
                all_pixel_values.append(pixel_values3)
                num_patches_lists.append([pixel_values1.size(0), pixel_values2.size(0), pixel_values3.size(0)])
                prompt = f"{question_}\n{pp_prompts[idx]}\n<image>\n<image>\n<image>"
            elif problem == 'object_localization':
                # For object_localization, use single image with PP
                all_pixel_values.append(pixel_values1)
                num_patches_lists.append([pixel_values1.size(0)])
                prompt = f"{question_}\n{pp_prompts[idx]}\n<image>"
            else:
                # For other problems, use two images
                all_pixel_values.append(pixel_values1)
                all_pixel_values.append(pixel_values2)
                num_patches_lists.append([pixel_values1.size(0), pixel_values2.size(0)])
                prompt = f"{question_}\n{pp_prompts[idx]}\n<image>\n<image>"
        else:
            # CoT mode
            # Add available images
            images = [pixel_values1]
            if 'pixel_values2' in locals() and pixel_values2 is not None:
                images.append(pixel_values2)
            if 'pixel_values3' in locals() and pixel_values3 is not None:
                images.append(pixel_values3)
                
            # Add images to batch
            for img in images:
                all_pixel_values.append(img)
                
            # Add patch sizes
            patch_sizes = [img.size(0) for img in images]
            num_patches_lists.append(patch_sizes)
            
            # Build prompt with appropriate number of image tags
            image_tags = ['<image>\n'] * len(images)
            prompt = ''.join(image_tags) + question_
        
        batch_prompts.append(prompt)
    
    # Concatenate all pixel values and move to device
    if len(all_pixel_values) > 0:
        all_pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16).cuda()
    else:
        all_pixel_values = None

    responses = batch_chat_multi_image(
        model,
        tokenizer,
        all_pixel_values,
        questions=batch_prompts,
        num_patches_lists=num_patches_lists,
        generation_config=generation_config,
        think_closing_tokens=think_closing_tokens
    )
    
    return responses

def extract_choice(text: str, problem='multi_view_reasoning'):
    """Extract the answer choice from model response"""
    # Get valid choices for this problem
    valid_choices = PROBLEM_CONFIGS[problem]['answer_choices']
    choices_pattern = ''.join(valid_choices) + ''.join([c.lower() for c in valid_choices])
    
    # First try to extract from \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^\}]+)\}', text)
    if boxed_match:
        content = boxed_match.group(1)
        # Extract letter from the boxed content
        letter_match = re.search(rf'\(?\s*([{choices_pattern}])\s*\)?', content)
        if letter_match:
            return f"({letter_match.group(1).upper()})"
    
    # Fallback to finding (A), (B), etc. pattern
    matches = re.findall(rf'(?<!\w)\(\s*([{choices_pattern}])\s*\)(?!\w)', text)
    return f"({matches[-1].upper()})" if matches else None


def run_internvl3(model, tokenizer, dataset, config, output_root, 
                  mode='pp', batch_size=1, problem='multi_view_reasoning',
                  model_size='2b',
                  max_new_tokens=MAX_NEW_TOKENS,
                  think_closing_tokens=FORCE_THINK_AFTER):
    """
    Run InternVL3 evaluation on BLINK dataset.
    
    Args:
        model: InternVL3 model
        tokenizer: Tokenizer
        dataset: BLINK dataset
        config: Configuration dict with paths to data files
        output_root: Output directory
        mode: 'pp', 'cot', or 'tool'
        batch_size: Batch size for inference
        problem: 'multi_view_reasoning' or 'visual_correspondence'
        model_size: Model size ('1b', '2b', or '4b')
        max_new_tokens: Token budget for generation
        think_closing_tokens: Number of tokens before forcing </think> (default: FORCE_THINK_AFTER)
    """
    if mode == 'pp':
        output_filename = f"internvl3_{model_size}_with_pp.jsonl"
        mode_desc = "PP"
    elif mode == 'tool':
        output_filename = f"internvl3_{model_size}_with_tool.jsonl"
        mode_desc = "Tool"
    else:
        output_filename = f"internvl3_{model_size}_cot.jsonl"
        mode_desc = "CoT"

    this_output_file = os.path.join(output_root, output_filename)

    # Initialize PP generator if using PP mode
    pp_generator = None
    jigsaw_stats = None
    semantic_stats = None
    obj_loc_annotations = None
    obj_loc_detections = None
    if mode == 'pp':
        print("Initializing perception program generator...")
        if problem == 'visual_correspondence':
            pp_generator = VisualCorrespondencePromptGenerator(
                correspondences_path=config['correspondences_path'],
                point_annotations_path=config['point_annotations_path'],
                task='visual_correspondence'
            )
            print(f"Loaded correspondences from: {config['correspondences_path']}")
            print(f"Loaded annotations from: {config['point_annotations_path']}")
        elif problem == 'jigsaw':
            # Load jigsaw stats for PP generation
            jigsaw_stats_path = config.get('jigsaw_stats_path', '')
            jigsaw_stats = load_jsonl(jigsaw_stats_path)
            print(f"Loaded jigsaw stats from: {jigsaw_stats_path}")
        elif problem == 'semantic_correspondence':
            # Load semantic correspondence stats for PP generation
            semantic_stats_path = config.get('semantic_stats_path', '')
            if semantic_stats_path:
                semantic_stats = load_jsonl(semantic_stats_path)
            else:
                semantic_stats = []
            print(f"Loaded semantic stats from: {semantic_stats_path}")
        elif problem == 'object_localization':
            # Load bounding box annotations
            annotations_path = config.get('annotations_path', '/blink_annotations/Object_Localization_val_annotations.jsonl')
            obj_loc_annotations = {}
            with open(annotations_path, 'r') as f:
                for line in f:
                    annot = json.loads(line)
                    obj_loc_annotations[annot['idx']] = annot
            print(f"Loaded annotations from: {annotations_path}")
            
            # Load detections
            detections_path = config.get('detections_path', 'saved/obj_localization/detections.jsonl')
            obj_loc_detections = {}
            with open(detections_path, 'r') as f:
                for line in f:
                    det = json.loads(line)
                    obj_loc_detections[det['idx']] = det
            print(f"Loaded detections from: {detections_path}")
        # For MVR, we'll generate PPs on-the-fly from flow data

    # Initialize tracking
    total_samples = len(dataset)
    all_indices = list(range(total_samples))
    results = []
    
    wrong_counter = 0
    total_time_st = time.time()

    # Setup generation config
    generation_config = {**GENERATION_CONFIG, 'max_new_tokens': max_new_tokens}

    # Create batches
    batches = []
    for i in range(0, total_samples, batch_size):
        batches.append(all_indices[i:i+batch_size])
    
    pbar = tqdm(batches, 
               desc=f"Evaluating {mode_desc} | Tokens={max_new_tokens}",
               leave=False, dynamic_ncols=True)
    
    for batch_indices in pbar:
        batch_samples = [dataset[i] for i in batch_indices]
        
        # Prepare mode-specific data for batch
        batch_pp_prompts = None
        batch_tool_imgs = None

        if mode == 'pp':
            # Generate perception programs for batch
            batch_pp_prompts = []
            for sample in batch_samples:
                img_id = sample['idx']
                
                if problem == 'visual_correspondence':
                    # Visual Correspondence: Use precomputed correspondences
                    question_text = sample['prompt'] + "\n" + sample['question']
                    full_pp_prompt = pp_generator.generate_prompt(question_text, img_id)
                    pp_only = full_pp_prompt[len(question_text):].strip()
                    batch_pp_prompts.append(pp_only)
                elif problem.startswith('hardblink'):
                    # Hardblink problems: PP is already preprocessed in the question field
                    # Extract full <perceptionprogram> blocks (including tags)
                    pp_match = re.findall(r'<perceptionprogram>.*?</perceptionprogram>', 
                                         sample['question'], flags=re.DOTALL)
                    if pp_match:
                        # Join all PP blocks (there might be multiple)
                        pp_str = '\n'.join(pp_match).strip()
                    else:
                        pp_str = ""
                    batch_pp_prompts.append(pp_str)
                elif problem == 'jigsaw':
                    # Jigsaw: Generate from precomputed stats
                    # Find the jigsaw stats entry for this image
                    jigsaw_field = None
                    for entry in jigsaw_stats:
                        if entry['idx'] == img_id:
                            jigsaw_field = entry
                            break
                    
                    if jigsaw_field:
                        jigsaw_pp, prog_json = pp.emit_perception_program(
                            modality="jigsaw",
                            field=jigsaw_field,
                            seg=None,
                            class_names=None,
                            grid=(10, 10),  # ignored for jigsaw
                            add_relations=False,  # not needed for jigsaw
                            tau=0.08,  # ignored for jigsaw
                            relation_cap=500  # ignored for jigsaw
                        )
                        batch_pp_prompts.append(jigsaw_pp)
                    else:
                        # If no stats found, append empty string
                        batch_pp_prompts.append("")
                elif problem == 'semantic_correspondence':
                    # Semantic Correspondence: Build combined PP (coords + semantic similarity)
                    sem_field = None
                    for entry in (semantic_stats or []):
                        if entry['idx'] == img_id:
                            sem_field = entry
                            break
                    if sem_field is not None:
                        # Build coords PP block for source point (Image 1) using utils helper
                        src_coords = sem_field.get('src_coords', None)
                        if src_coords and len(src_coords) == 2:
                            coords_pp = coords_to_perceptionprogram_multi_image([
                                {
                                    "x": src_coords[0],
                                    "y": src_coords[1],
                                    "label": "REF",
                                    "image": "Image 1 (Source)"
                                }
                            ])
                        else:
                            coords_pp = ""

                        semantic_pp, _ = pp.emit_perception_program(
                            modality="semantic",
                            field=sem_field,
                            seg=None,
                            class_names=None,
                            grid=(10, 10),
                            add_relations=False,
                            tau=0.08,
                            relation_cap=500
                        )
                        batch_pp_prompts.append(f"{coords_pp}\n{semantic_pp}".strip())
                    else:
                        batch_pp_prompts.append("")
                elif problem == 'object_localization':
                    # Object Localization: Generate two PPs - one for bounding boxes, one for detections
                    combined_pp = ""
                    
                    # First PP: Bounding box coordinates
                    if obj_loc_annotations and img_id in obj_loc_annotations:
                        annot = obj_loc_annotations[img_id]
                        coords = annot.get('coords', [])
                        
                        # Convert label names to more descriptive format
                        formatted_coords = []
                        for coord in coords:
                            label = coord['label']
                            # Convert "A_upper_left" to "Bounding box A, upper-left corner"
                            if '_' in label:
                                bbox_id, position = label.split('_', 1)
                                position_formatted = position.replace('_', '-')
                                new_label = f"Bounding box {bbox_id}, {position_formatted} corner"
                            else:
                                new_label = label
                            
                            formatted_coords.append({
                                'x': int(round(coord['x'])),
                                'y': int(round(coord['y'])),
                                'label': new_label
                            })
                        
                        # Use coords_to_perceptionprogram from utils
                        bbox_pp = coords_to_perceptionprogram(formatted_coords)
                        combined_pp = bbox_pp
                    
                    # Second PP: Detection results
                    if obj_loc_detections and img_id in obj_loc_detections:
                        det_data = obj_loc_detections[img_id]
                        detections = det_data.get('detections', [])
                        # Generate PP from detections
                        det_pp, _ = pp.emit_perception_program(
                            modality="object-detection",
                            field=detections,
                            seg=None,
                            class_names=None,
                            grid=(10, 10),
                            add_relations=False,
                            tau=0.08,
                            relation_cap=500
                        )
                        combined_pp = f"{combined_pp}\n\n{det_pp}" if combined_pp else det_pp
                    
                    batch_pp_prompts.append(combined_pp)
                else:
                    # MVR: Generate from optical flow
                    flow_path = config.get('flow_path', '')
                    flow_array = np.load(f"{flow_path}{img_id}.npy")
                    flow_pp, prog_json = pp.emit_perception_program(
                        modality="flow",
                        field=flow_array,
                        seg=None,
                        class_names=None,
                        grid=(10, 10),
                        add_relations=False,
                        tau=0.08,
                        relation_cap=500)
                    batch_pp_prompts.append(flow_pp)
                
        elif mode == 'tool':
            # Load tool visualization images for batch
            batch_tool_imgs = []
            for sample in batch_samples:
                img_id = sample['idx']
                tool_path = config.get('raw_tool_path', config.get('tool_path', ''))
                if problem == 'jigsaw':
                    # For jigsaw, load TWO stitched images
                    stitched_img1_path = os.path.join(tool_path, f"{img_id}_img1.png")
                    stitched_img2_path = os.path.join(tool_path, f"{img_id}_img2.png")
                    stitched_img1 = Image.open(stitched_img1_path)
                    stitched_img2 = Image.open(stitched_img2_path)
                    # Append as a tuple of two images
                    batch_tool_imgs.append((stitched_img1, stitched_img2))
                elif problem.startswith('hardblink'):
                    # For hardblink, load depth map
                    tool_img_path = os.path.join(tool_path, f"{img_id}.png")
                    tool_img = Image.open(tool_img_path)
                    batch_tool_imgs.append(tool_img)
                elif problem == 'object_localization':
                    # For object_localization, load visualization with bounding boxes
                    visualizations_path = config.get('visualizations_path', 'saved/obj_localization/visualizations/')
                    tool_img_path = os.path.join(visualizations_path, f"{img_id}.png")
                    tool_img = Image.open(tool_img_path)
                    batch_tool_imgs.append(tool_img)
                elif problem == 'multi_view_reasoning':
                    tool_img_path = os.path.join(tool_path, f"{img_id}.png")
                    tool_img = Image.open(tool_img_path)
                    batch_tool_imgs.append(tool_img)
                elif problem == 'semantic_correspondence':
                    # For semantic correspondence, use *_tool.png
                    tool_img_path = os.path.join(tool_path, f"{img_id}_tool.png")
                    tool_img = Image.open(tool_img_path)
                    batch_tool_imgs.append(tool_img)
                else:  # visual_correspondence
                    tool_img_path = os.path.join(tool_path, f"{img_id}.png")
                    tool_img = Image.open(tool_img_path)
                    batch_tool_imgs.append(tool_img)

        # Get tool template from config if in tool mode
        tool_template = config.get('template', None) if mode == 'tool' else None

        st = time.time()
        # Always use batch processing (single samples are just batch size 1)
        outputs = call_internvl3_batch(model, tokenizer, batch_samples, batch_pp_prompts,
                                      batch_tool_imgs, tool_template, generation_config=generation_config, 
                                      problem=problem, think_closing_tokens=think_closing_tokens)
        end = time.time()

        batch_time = end - st

        # Process results for each sample in batch
        for local_idx, (ds_idx, sample, output) in enumerate(zip(batch_indices, batch_samples, outputs)):
            prediction = extract_choice(output, problem=problem)
            
            # Count actual tokens in the output
            output_tokens = tokenizer.encode(output, add_special_tokens=False)
            actual_token_count = len(output_tokens)
            
            result = {
                'image': sample['idx'],
                'question': sample["prompt"] + "\n" + sample['question'],
                'ground_truth': sample['answer'],
                'prediction': prediction if prediction else "",
                'output': output,
                'correct': (prediction == sample['answer']) if prediction else False,
                'max_new_tokens': max_new_tokens,
                'actual_output_tokens': actual_token_count,
            }


            results.append(result)
            
            if not prediction:
                pass  # No valid prediction
            else:
                wrong_counter += (not result["correct"])

        # Calculate running accuracy
        completed_results = [r for r in results if r['prediction'] != '']
        correct_so_far = sum(1 for r in completed_results if r['correct'])
        total_completed = len(completed_results)
        accuracy_so_far = correct_so_far / total_completed if total_completed > 0 else 0.0
        
        pbar.set_postfix({
            "Acc": f"{accuracy_so_far:.3f}",
            "Wrong": wrong_counter,
            "Time": f"{batch_time:.2f}s"
        }, refresh=True)

    total_time_end = time.time()

    # Collect final results
    correct = sum(r["correct"] for r in results)
    acc = correct / len(results) if results else 0.0
    unanswered = len([r for r in results if r['prediction'] == ''])
    
    print(f"\nModel: InternVL3-{model_size.upper()} ({mode_desc}) | Problem: {problem}")
    print(f"Final Accuracy: {acc:.4f} ({correct}/{len(results)})")
    print(f"Unanswered: {unanswered}")
    print(f"Total Time Taken: {total_time_end - total_time_st:.2f}s")

    # Write results
    with open(this_output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Wrote {len(results)} results → {this_output_file}")
    
    return acc

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate InternVL3 on BLINK Benchmark')
    parser.add_argument('--problem', type=str, 
                        choices=['multi_view_reasoning', 'visual_correspondence', 
                                'hardblink3pointscenter', 'hardblink4pointscenter', 'hardblink5pointscenter',
                                'jigsaw', 'semantic_correspondence', 'object_localization'],
                        default='multi_view_reasoning',
                        help='Problem type to evaluate (default: multi_view_reasoning)')
    parser.add_argument('--mode', type=str, choices=['pp', 'cot', 'tool'], default='pp',
                        help='Evaluation mode: "pp" for perception programs, "cot" for chain-of-thought, "tool" for visualization (default: pp)')
    parser.add_argument('--model_size', type=str, choices=['1b', '2b', '4b'], default='2b',
                        help='Model size: "1b", "2b" or "4b" (default: 2b)')
    parser.add_argument('--subsample', type=int, default=SUBSAMPLE_SIZE,
                        help=f'Number of samples for prototyping (default: {SUBSAMPLE_SIZE}, None = full dataset)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for inference (default: {BATCH_SIZE})')
    parser.add_argument('--think_closing_tokens', type=int, default=FORCE_THINK_AFTER,
                        help='Number of tokens to generate before forcing </think> tag (default: FORCE_THINK_AFTER)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file (default: auto-select based on problem and mode)')
    args = parser.parse_args()

    # Validate problem
    if args.problem not in PROBLEM_CONFIGS:
        raise ValueError(f"Unknown problem: {args.problem}")

    # Load configuration
    if args.config is None:
        # Auto-select config based on problem and mode
        if args.mode == 'pp':
            config_path = f'eval_config/pp_config/{args.problem}.json'
        elif args.mode == 'tool':
            config_path = f'eval_config/raw_tool_config/{args.problem}.json'
        elif args.mode == 'cot' and args.problem.startswith('hardblink'):
            # CoT mode needs config for hardblink problems
            config_path = f'eval_config/cot_config/{args.problem}.json'
        else:
            config = {}  # CoT mode doesn't need config for non-hardblink problems
            config_path = None
    else:
        config_path = args.config
    
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        config = {}

    # Select model based on size
    if args.model_size == '1b':
        model_path = 'OpenGVLab/InternVL3_5-1B-MPO'
    elif args.model_size == '2b':
        model_path = 'OpenGVLab/InternVL3_5-2B-MPO'
    else:  # 4b
        model_path = 'OpenGVLab/InternVL3_5-4B-MPO'
    
    # Set output directory based on problem
    output_root = f"./output/{args.problem}"
    
    # Load dataset
    dataset_name = PROBLEM_CONFIGS[args.problem]['dataset_key']
    
    # Check if this is a hardblink problem (custom JSONL dataset)
    if args.problem.startswith('hardblink'):
        # Load paths from config
        if not config:
            raise ValueError("Config file required for hardblink problems. Expected config but got none.")
        
        jsonl_path = config.get('jsonl_path')
        image_dir = config.get('image_dir')
        
        if not jsonl_path or not image_dir:
            raise ValueError("Config must contain 'jsonl_path' and 'image_dir' for hardblink problems")
        
        print(f"Loading custom dataset from: {jsonl_path}")
        dataset = load_custom_jsonl_dataset(jsonl_path, image_dir)
        
        # Subsample if requested
        if args.subsample is not None:
            print(f"Subsampling {args.subsample} samples from {len(dataset)} total samples")
            dataset = dataset[:min(args.subsample, len(dataset))]
    else:
        # Load from BLINK dataset
        print(f"Loading BLINK {dataset_name} dataset...")
        dataset = load_dataset("BLINK-Benchmark/BLINK", dataset_name,
                              cache_dir="/mllm_datasets/BLINK")["val"]
        
        # Subsample if requested
        if args.subsample is not None:
            print(f"Subsampling {args.subsample} samples from {len(dataset)} total samples")
            dataset = dataset.select(range(min(args.subsample, len(dataset))))
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Problem: {args.problem}")
    print(f"Model size: {args.model_size.upper()}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Batch size: {args.batch_size}")
    
    # Setup model
    print(f"Loading InternVL3-{args.model_size.upper()} model...")
    model, tokenizer = setup_model(model_path, mode=args.mode, problem=args.problem)
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Run evaluation
    accuracy = run_internvl3(model, tokenizer, dataset, config,
                            output_root, mode=args.mode, batch_size=args.batch_size, 
                            problem=args.problem, model_size=args.model_size,
                            think_closing_tokens=args.think_closing_tokens)
    
    print(f"\nFinal Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()

