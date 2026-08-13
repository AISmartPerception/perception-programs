"""
    Code to run GPT-5 on Multi-View Reasoning from BLINK.
    Three methods are provided: vanilla (Standard), tool (Tool), pp (Perception Program) 
"""
from datasets import load_dataset
import random
import torch
import numpy as np
import generate_perception_program as pp
from tqdm import tqdm
import json, os, time
from io import BytesIO
from openai import OpenAI
from PIL import Image
import base64
from torchvision.transforms.functional import pil_to_tensor

dataset = load_dataset("BLINK-Benchmark/BLINK", "Multi-view_Reasoning",
                       cache_dir="/mllm_datasets/BLINK")["val"]
random.seed(42)
dataset_size = len(dataset)

OPENAI_KEY = "" # paste the key here (or put it as env variable and load it)
client = OpenAI(api_key=OPENAI_KEY)

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

def _to_data_uri(image: Image.Image, format: str = None) -> str:
    image = resize_image_max_dim(image)
    if format is None:
        format = image.format or "PNG"
    
    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp"
    }
    
    format = format.upper()
    mime = mime_types.get(format, "image/png")
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    
    return f"data:{mime};base64,{b64}"

def call_gpt_vanilla(question, model="gpt-5-mini-2025-08-07"):
    system =  \
    """
        You are an oracle answering multiple-choice questions about images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    # these are already an PIL images
    img1 = question["image_1"]
    img2 = question['image_2']
    prompt = question['prompt'] + "\n" + question["question"]
    im1_uri = _to_data_uri(img1)
    im2_uri = _to_data_uri(img2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": im1_uri}},
                    {"type": "image_url", "image_url": {"url": im2_uri}},
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ],
        max_completion_tokens=8192,
    )
    return response

def run_gpt_vanilla(output_root):
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gpt5mini.jsonl")

    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Processing Multi-View Reasoning")
    updated = False
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
                output = call_gpt_vanilla(sample, model="gpt-5-mini-2025-08-07")
                prediction = output.choices[0].message.content.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage.prompt_tokens,
                    'response_tokens': output.usage.completion_tokens,
                    'total_tokens': output.usage.total_tokens
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

def call_gpt_with_pp(question, flow_pp, model):
    system = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. When answering, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    # these are already an PIL images
    img1 = question["image_1"]
    img2 = question['image_2']
    question_ = question['prompt'] + "\n" + question["question"]
    prompt = '\n'.join([question_, flow_pp])
    
    im1_uri = _to_data_uri(img1)
    im2_uri = _to_data_uri(img2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": im1_uri}},
                    {"type": "image_url", "image_url": {"url": im2_uri}},
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ],
        max_completion_tokens=8192,
    )
    return response

def run_gpt_with_pp(output_root):
    flow_path = "/multiview_reasoning/flow_data/"
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gpt5mini.jsonl")

    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Multiview Reasoning w/ PerceptionProgram!")
    updated = False

    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        flow_array = np.load(f"{flow_path}{img_id}.npy")
        flow_pp, prog_json = pp.emit_perception_program(
            modality="flow",
            field=flow_array,
            seg=None,
            class_names=None,
            grid=(10,10),
            add_relations=False, # not needed for flow
            tau=0.08, # ignored for flow
            relation_cap=500)

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
                output = call_gpt_with_pp(sample, flow_pp,
                                          model="gpt-5-mini-2025-08-07")
                prediction = output.choices[0].message.content.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage.prompt_tokens,
                    'response_tokens': output.usage.completion_tokens,
                    'total_tokens': output.usage.total_tokens
                }

                existing[img_id] = result
                success = True
                updated = True

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

def call_gpt_with_raft(question, raft_flow, model):
    system =  \
    """
        You are an oracle answering multiple-choice questions about images. You will be provided two images (image 1 and image 2) and their corresponding optical flow, you should use both to answer. In the optical flow, purple, pink, red, yellow, orange hues indicate rightward motion, while blue, turquoise, green, cyan indicate leftward motion. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    
    img1 = question["image_1"]
    img2 = question['image_2']
    prompt = question['prompt'] + "\n" + question["question"]
    im1_uri = _to_data_uri(img1)
    im2_uri = _to_data_uri(img2)
    raft_uri = _to_data_uri(raft_flow)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1".strip()},
                    {"type": "image_url", "image_url": {"url": im1_uri}},
                    {"type": "text", "text": "Image 2".strip()},
                    {"type": "image_url", "image_url": {"url": im2_uri}},
                    {"type": "text", "text": "Optical Flow".strip()},
                    {"type": "image_url", "image_url": {"url": raft_uri}},
                    {"type": "text", "text": prompt.strip()},
                ],
            },
        ],
        max_completion_tokens=8192,
    )
    return response

def run_gpt_with_tool(output_root):
    flow_path = "/multiview_reasoning/flow_data/imgs/"
    max_retries = 3
    this_output_file = os.path.join(output_root, f"gpt5mini.jsonl")

    results = []
    existing = {r['image']: r for r in results}

    pbar = tqdm(dataset, desc=f"Multiview Reasoning w/ PP!")
    updated = False 

    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        flow_img = Image.open(f"{flow_path}{img_id}.png")
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
                output = call_gpt_with_raft(sample, flow_img,
                                          model="gpt-5-mini-2025-08-07")
                prediction = output.choices[0].message.content.strip()
                result = {
                    'image': sample['idx'],
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'correct': prediction == sample['answer'],
                    'prompt_tokens': output.usage.prompt_tokens,
                    'response_tokens': output.usage.completion_tokens,
                    'total_tokens': output.usage.total_tokens
                }

                existing[img_id] = result
                success = True
                updated = True

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
    output_root = "/multiview_reasoning/gpt5mini/vanilla/"
    run_gpt_vanilla(output_root)

    output_root = "/multiview_reasoning/gpt5mini/pp/"
    run_gpt_with_pp(output_root)

    output_root = "/multiview_reasoning/gpt5mini/tool/"
    run_gpt_with_tool(output_root)