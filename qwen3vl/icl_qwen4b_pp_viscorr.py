from datasets import load_dataset
import random, json, os, time, re, cv2, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import kornia as K
from kornia.feature import LoFTR
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import generate_perception_program as pp

mode = "Thinking"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
dataset_size = len(dataset)
root = "/data00/visual_correspondence/"
matcher = LoFTR(pretrained='outdoor')
model_name = f"Qwen/Qwen3-VL-4B-{mode}"
# keep this same
target_dir = "/data00/multiview_reasoning/open-source-models/"
processor = AutoProcessor.from_pretrained(f"Qwen/Qwen3-VL-4B-{mode}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=target_dir)

def call_qwen3_4b_pp(question, coords_pp, correspondence_pp):
    system = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. When answering, only output the letter enclosed by parenthesis---i.e. (A)---without any prose. 
    """
    icl_example = """
        You're given two images of the same scene and one or more <perceptionprogram> blocks.
        The task: find which red-circled point in IMAGE 2 (options A/B/C/D) corresponds to the single
        reference point (REF) in IMAGE 1.

        One example of how to use perception program to solve the problem is given below.
                
        Images: (reference, target)
        PP (excerpt): REF in image1 at (546,795); image2 candidates:
        A(46,68) B(142,849) C(547,797) D(765,186); correspondences show many c≈r.
        Model output:
        Justification: PP: REF (546,795); C is nearest (Δ≈2 px); correspondences indicate minimal shift.
        Answer: (C)

        Now use this and solve the given question.
    """
    
    img1 = question["image_1"]
    img2 = question['image_2']
    question = question['prompt'] + "\n" + question["question"]
    prompt = '\n'.join([question, coords_pp, correspondence_pp])
    system_prompt = '\n'.join([system, icl_example])

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt")
    inputs = inputs.to(model.device)
    # by default do_sample=False, so greedy decoding -- set it too for safety
    generated_ids = model.generate(**inputs, do_sample=False, 
                                   max_new_tokens=32768)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return output_text

def extract_choice(text: str):
    matches = re.findall(r'(?<!\w)\(\s*([ABCDEabcde])\s*\)(?!\w)', text)
    return f"({matches[-1].upper()})" if matches else ""

def load_annotations(annotations_path):
    """Load annotations from JSONL file indexed by dataset idx."""
    data_by_idx = {}
    with open(annotations_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["idx"] not in data_by_idx:
                    data_by_idx[record["idx"]] = {}
                data_by_idx[record["idx"]][record["image_field"]] = record
    return data_by_idx

def coords_to_perceptionprogram(coords_imgA, coords_imgB):
    """
        Convert coords list into <perceptionprogram> YAML-like block.
    """
    lines = []
    lines.append("<perceptionprogram>")
    lines.append("modality: point-detection")
    lines.append("granularity: pixels")
    lines.append('note: "Each item lists one of the points relative to this task in the following format:')
    lines.append("- c: coordinates of the point")
    lines.append("- b: label of the point")
    lines.append("- i: name of the image the point belongs to")
    lines.append('Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right."')    
    lines.append("items:")
    
    coords = {"image 1 (reference)": coords_imgA, "image 2 (target)": coords_imgB}
    for k in coords.keys():
        for pt in coords[k]:
            x = int(round(pt["x"]))
            y = int(round(pt["y"]))
            label = pt["label"]
            lines.append(f"  - c: [{x}, {y}]")
            lines.append(f"  - b: \"{label}\"")
            lines.append(f"  - i: \"{k}\"")

    lines.append("</perceptionprogram>")
    return "\n".join(lines)

def compute_correspondences(img1, img2):
    confidence_threshold = 0.3
    max_matches = 100

    im1_t = pil_to_tensor(img1).to(dtype=torch.float32)/255.0
    im2_t = pil_to_tensor(img2).to(dtype=torch.float32)/255.0
    im1_t = im1_t.unsqueeze(0)
    im2_t = im2_t.unsqueeze(0)
    
    input_dict = {"image0": K.color.rgb_to_grayscale(im1_t),
                  "image1": K.color.rgb_to_grayscale(im2_t)}
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    kpts0 = correspondences['keypoints0'].cpu().numpy()
    kpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    
    mask = conf >= confidence_threshold
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]
    conf = conf[mask]

    if len(kpts0) > max_matches:
        indices = np.random.choice(len(kpts0), max_matches, replace=False)
        kpts0 = kpts0[indices]
        kpts1 = kpts1[indices]
        conf = conf[indices]
    
    return {"kpts0": kpts0, "kpts1": kpts1,
            "W1": img1.width, "H1": img1.height,
            "W2": img2.width, "H2": img2.height}

def run_qwen_with_pp(output_root):
    pbar = tqdm(dataset, desc=f"Processing VisCorr!")
    this_output_file = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")
    existing = {}
    annotations = load_annotations("/data00/visual_correspondence/Visual_Correspondence_val_annotations.jsonl")
    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        st = time.time()
        coords_pp = coords_to_perceptionprogram(annotations[img_id]['image_1']['coords'],
                                                annotations[img_id]['image_2']['coords'])
        correspondence_vals = compute_correspondences(sample['image_1'], 
                                                      sample['image_2'])
        correspondence_pp, prog_json = pp.emit_perception_program(
            modality="correspondence",
            field=correspondence_vals,
            seg=None,
            class_names=None,
            grid=(10,10), # ignored for correspondence
            add_relations=False, # not needed for correspondence
            tau=0.08, # ignored for correspondence
            relation_cap=100)

        output = call_qwen3_4b_pp(sample, coords_pp, correspondence_pp)
        end = time.time()
        prediction = extract_choice(' '.join(output))
        result = {
            'image': sample['idx'],
            'question': sample['question'],
            'ground_truth': sample['answer'],
            'prediction': prediction,
            'output': output,
            'correct': prediction == sample['answer'],
        }
        pbar.set_postfix({
            'GT': sample['answer'],
            'Pred': prediction[:15],
            'Correct': result['correct'],
            'Time': f"{end-st:.2f}s"
        }, refresh=True)
        existing[img_id] = result

    results = list(existing.values())
    with open(this_output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    dataset_results = results
    correct_predictions = sum(1 for r in dataset_results if r.get('correct', False))
    total_predictions = len(dataset_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Model: {'-'.join(output_root.split('/')[-3:-1]).title()} | Accuracy: {accuracy}")

if __name__ == "__main__":
    output_root = f"/data00/visual_correspondence/open-source-models/qwen4b-{mode.lower()}/pp/"
    run_qwen_with_pp(output_root)