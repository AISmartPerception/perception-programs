"""
    Code to run Gemini 2.5 Pro on Semantic Correspondence from BLINK.
    Three methods are provided: vanilla (Standard), tool (Tool), pp (Perception Program) 
"""
from datasets import load_dataset
import random
import numpy as np
import generate_perception_program as pp
from tqdm import tqdm
import json, os, time
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image

# load dataset
dataset = load_dataset("BLINK-Benchmark/BLINK", "Semantic_Correspondence",
                       cache_dir="/mllm_datasets/BLINK")["val"]
random.seed(42)
dataset_size = len(dataset)
client = genai.Client(api_key="") # paste Gemini key here

def resize_image_max_dim(img: Image.Image, max_dim: int = 336) -> Image.Image:
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        return img
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def _pil_image_to_bytes(img: Image.Image):
    """Convert PIL Image to bytes and determine MIME type."""
    img = resize_image_max_dim(img)
    # Get format from PIL Image (e.g., 'JPEG', 'PNG', 'WEBP')
    img_format = img.format if img.format else 'PNG'
    
    format_to_mime = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    mime = format_to_mime.get(img_format.upper(), "image/png")
    buffer = BytesIO()
    save_format = img_format if img_format in format_to_mime else 'PNG'
    img.save(buffer, format=save_format)
    img_bytes = buffer.getvalue()
    return img_bytes, mime

def call_gemini(question,
                model="gemini-2.5-pro"):
    system = """
        You are an oracle answering multiple-choice questions about images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """

    img1 = question['image_1']
    img2 = question['image_2']

    prompt = question['prompt'] + "\n" + question["question"]
    img1_bytes, img1_mime = _pil_image_to_bytes(img1)
    img2_bytes, img2_mime = _pil_image_to_bytes(img2)

    contents = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=img1_bytes, mime_type=img1_mime),
            types.Part.from_bytes(data=img2_bytes, mime_type=img2_mime),
            types.Part.from_text(text=prompt),
        ],
    )

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=8000,),
        )
    return resp

def run_gemini_vanilla(output_root):
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gemini25_pro.jsonl")
    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Processing SemanticCorr.")
    updated = False  # track if we need to rewrite JSONL fully
    for i, sample in enumerate(pbar):
        img_id = sample['idx']

        if img_id in existing and "error" not in existing[img_id]:
            prev_result = existing[img_id]
            pbar.set_postfix({
                'GT': prev_result['ground_truth'],
                'Pred': (prev_result['prediction'] or "")[:15],
                'Correct': prev_result['correct']
            }, refresh=True)
            continue

        success = False
        for attempt in range(max_retries):
            try:
                output = call_gemini(sample, model="gemini-2.5-pro")
                prediction = output.text.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage_metadata.prompt_token_count,
                    'response_tokens': output.usage_metadata.candidates_token_count,
                    'total_tokens': output.usage_metadata.total_token_count
                }

                existing[img_id] = result
                success = True
                updated = True

                # tqdm info
                pbar.set_postfix({
                    'GT': sample['answer'],
                    'Pred': prediction[:15],
                    'Correct': result['correct']
                }, refresh=True)
                break

            except Exception as e:
                print(f"  Error on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(1)  # small delay before retry

        if not success:
            existing[img_id] = {
                'image': sample['idx'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'prediction': None,
                'correct': False,
                'error': f"Failed after {max_retries} attempts"
            }
            updated = True

    results = list(existing.values())
    if updated:
        with open(this_output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    dataset_results = results
    correct_predictions = sum(1 for r in dataset_results if r.get('correct', False))
    total_predictions = len(dataset_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    total_tokens_used = sum(r.get('total_tokens', 0) for r in dataset_results if 'total_tokens' in r)
    avg_tokens = total_tokens_used / len([r for r in dataset_results if 'total_tokens' in r]) if dataset_results else 0

    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total tokens used: {total_tokens_used}")
    print(f"Average tokens per sample: {avg_tokens:.1f}")

def call_gemini_with_pp(question, coords_pp, semantic_pp, model):
    system = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. If all scores are very low in <perceptionprogram>, look at the images and make your best guess. When answering, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    img1 = question['image_1']
    img2 = question['image_2']

    question_ = question['prompt'] + "\n" + question["question"]
    prompt = '\n'.join([question_, coords_pp, semantic_pp])
    img1_bytes, img1_mime = _pil_image_to_bytes(img1)
    img2_bytes, img2_mime = _pil_image_to_bytes(img2)

    contents = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=img1_bytes, mime_type=img1_mime),
            types.Part.from_bytes(data=img2_bytes, mime_type=img2_mime),
            types.Part.from_text(text=prompt),
        ],
    )

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=8000,),
        )
    return resp

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def coords_to_perceptionprogram(coords):
    lines = []
    lines.append("<perceptionprogram>")
    lines.append("modality: point-detection")
    lines.append("granularity: pixels")
    lines.append('note: "Each item lists one of the points relative to this task in the following format:')
    lines.append("- c: coordinates of the point")
    lines.append("- i: name of the image the point belongs to")
    lines.append('Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right."')    
    lines.append("items:")
    
    lines.append(f"  - c: [{coords['src_coords'][0]}, {coords['src_coords'][1]}]")
    lines.append(f"  - i: \"Image 1 (Source)\"")
    lines.append("</perceptionprogram>")
    return "\n".join(lines)

def run_gemini_with_pp(output_root):
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gemini25_pro.jsonl")
    pp_outputs = read_jsonl("/semantic_correspondence/raw_pp_outputs_2.jsonl")
    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Processing SemanticCorr w/ Perception Program!")
    updated = False 
    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        pp_val = pp_outputs[i]
        assert pp_val['idx'] == img_id
        
        semantic_pp, prog_json = pp.emit_perception_program(
            modality="semantic",
            field=pp_val,
            seg=None,
            class_names=None,
            grid=(10,10), # not needed for similarity
            add_relations=False, # not needed for similarity
            tau=0.08, # not needed for similarity
            relation_cap=500 # not needed for similarity
        )
        coords_pp = coords_to_perceptionprogram(pp_val)

        if img_id in existing and "error" not in existing[img_id]:
            prev_result = existing[img_id]
            pbar.set_postfix({
                'GT': prev_result['ground_truth'],
                'Pred': (prev_result['prediction'] or "")[:15],
                'Correct': prev_result['correct']
            }, refresh=True)
            continue

        success = False
        for attempt in range(max_retries):
            try:
                output = call_gemini_with_pp(sample, coords_pp,
                                             semantic_pp,
                                             model="gemini-2.5-pro")
                prediction = output.text.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage_metadata.prompt_token_count,
                    'response_tokens': output.usage_metadata.candidates_token_count,
                    'total_tokens': output.usage_metadata.total_token_count
                }

                existing[img_id] = result
                success = True
                updated = True

                # tqdm info
                pbar.set_postfix({
                    'GT': sample['answer'],
                    'Pred': prediction[:15],
                    'Correct': result['correct']
                }, refresh=True)
                break

            except Exception as e:
                print(f"  Error on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(1)

        if not success:
            existing[img_id] = {
                'image': sample['idx'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'prediction': None,
                'correct': False,
                'error': f"Failed after {max_retries} attempts"
            }
            updated = True

    results = list(existing.values())
    if updated:
        with open(this_output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    dataset_results = results
    correct_predictions = sum(1 for r in dataset_results if r.get('correct', False))
    total_predictions = len(dataset_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    total_tokens_used = sum(r.get('total_tokens', 0) for r in dataset_results if 'total_tokens' in r)
    avg_tokens = total_tokens_used / len([r for r in dataset_results if 'total_tokens' in r]) if dataset_results else 0

    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total tokens used: {total_tokens_used}")
    print(f"Average tokens per sample: {avg_tokens:.1f}")

def call_gemini_with_tool(question, semantic_map_out, model):
    system =  \
    """
        You are an oracle answering multiple-choice questions about images. You will be provided two images (source and target) and corresponding similarity density map overlayed on target image. In density map, brighter colors indicate higher similarity. Use the similarity map and images to answer. In the visual correspondence map (left image is image 1, right image is image 2), each line connects two similar points in the two images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    img1 = question["image_1"]
    img2 = question['image_2']
    prompt = question['prompt'] + "\n" + question["question"]
    img1_bytes, img1_mime = _pil_image_to_bytes(img1)
    img2_bytes, img2_mime = _pil_image_to_bytes(img2)
    sim_map_bytes, sim_map_mime = _pil_image_to_bytes(semantic_map_out)

    contents = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=img1_bytes, mime_type=img1_mime),
            types.Part.from_bytes(data=img2_bytes, mime_type=img2_mime),
            types.Part.from_bytes(data=sim_map_bytes, mime_type=sim_map_mime),
            types.Part.from_text(text=prompt),
        ],
    )

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=8000,),
        )
    return resp

def run_gemini_with_tool(output_root):
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gemini25_pro.jsonl")
    tool_outputs = "/semantic_correspondence/tool_out_imgs/"
    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Processing SemanticCorr w/ Tool!")
    updated = False  # track if we need to rewrite JSONL fully
    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        tool_output_img = f"{tool_outputs}{img_id}_tool.png"
        tool_output_img_read = Image.open(tool_output_img).convert('RGB')

        if img_id in existing and "error" not in existing[img_id]:
            prev_result = existing[img_id]
            pbar.set_postfix({
                'GT': prev_result['ground_truth'],
                'Pred': (prev_result['prediction'] or "")[:15],
                'Correct': prev_result['correct']
            }, refresh=True)
            continue

        success = False
        for attempt in range(max_retries):
            try:
                output = call_gemini_with_tool(sample, tool_output_img_read,
                                               model="gemini-2.5-pro")
                prediction = output.text.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage_metadata.prompt_token_count,
                    'response_tokens': output.usage_metadata.candidates_token_count,
                    'total_tokens': output.usage_metadata.total_token_count
                }

                existing[img_id] = result
                success = True
                updated = True

                # tqdm info
                pbar.set_postfix({
                    'GT': sample['answer'],
                    'Pred': prediction[:15],
                    'Correct': result['correct']
                }, refresh=True)
                break

            except Exception as e:
                print(f"  Error on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(1)

        if not success:
            existing[img_id] = {
                'image': sample['idx'],
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'prediction': None,
                'correct': False,
                'error': f"Failed after {max_retries} attempts"
            }
            updated = True

    results = list(existing.values())
    if updated:
        with open(this_output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    dataset_results = results
    correct_predictions = sum(1 for r in dataset_results if r.get('correct', False))
    total_predictions = len(dataset_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    total_tokens_used = sum(r.get('total_tokens', 0) for r in dataset_results if 'total_tokens' in r)
    avg_tokens = total_tokens_used / len([r for r in dataset_results if 'total_tokens' in r]) if dataset_results else 0

    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total tokens used: {total_tokens_used}")
    print(f"Average tokens per sample: {avg_tokens:.1f}")

if __name__ == "__main__":
    output_root = "/semantic_correspondence/frontier_models/gemini25pro/vanilla/"
    os.makedirs(output_root, exist_ok=True)
    run_gemini_vanilla(output_root)

    output_root = "/semantic_correspondence/frontier_models/gemini25pro/pp/"
    os.makedirs(output_root, exist_ok=True)
    run_gemini_with_pp(output_root)

    output_root = "/semantic_correspondence/frontier_models/gemini25pro/tool/"
    os.makedirs(output_root, exist_ok=True)
    run_gemini_with_tool(output_root)