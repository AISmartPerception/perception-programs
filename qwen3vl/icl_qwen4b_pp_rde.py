import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch, jsonlines, cv2
import generate_perception_program as pp
import numpy as np
from PIL import Image

mode = "Thinking"  # or "Instruct"
random.seed(42)
torch.manual_seed(42)

model_name = f"Qwen/Qwen3-VL-4B-{mode}"
target_dir = "/data00/multiview_reasoning/open-source-models/"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 8192
MAX_TRIES = 1
RETRY_NEW_TOKENS_FACTOR = 1.0

processor = AutoProcessor.from_pretrained(model_name, cache_dir=target_dir)
processor.tokenizer.padding_side = "left"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=target_dir,
).eval()

def extract_choice(text: str):
    m = re.findall(r'(?<!\w)\(\s*([ABCDEabcde])\s*\)(?!\w)', text)
    return f"({m[-1].upper()})" if m else ""

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_point_positions(jsonl_path):
    """Load JSONL into a dict keyed by image filename."""
    lookup = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            lookup[entry["image"]] = entry["coords"]
    return lookup

def coords_to_perceptionprogram(coords):
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
    lines.append('Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right."')
    lines.append("items:")

    for pt in coords:
        x = int(round(pt["x"]))
        y = int(round(pt["y"]))
        label = pt["label"]
        lines.append(f"  - c: ({x}, {y})")
        lines.append(f"    b: \"{label}\"")

    lines.append("</perceptionprogram>")
    return "\n".join(lines)

def build_conversation(sample, k, depth_pp):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. When answering, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
        Output Rules:
            - Start with a short thinking section enclosed by <think> and </think> tags.
            - Provide a single final choice as (X) where X is A, B, C, etc.
            - Include, before the final choice, one short justification (maximum 20 words).
            - Do not include extra thoughts, <think> tags, step-by-step reasoning, chain-of-thought reasoning, etc.
    """
    icl_3p_example = """
        You are given an image with multiple points circled on, and a <perceptionprogram> block. Using the image and <perceptionprogram> decide which point is the closest to the camera?

        One demonstrative example of how to use perception program to solve a certain problem is given below.

        Image: (scene with three labeled points A, B, C)
        Perception (depth in [0,1], higher = closer):
        - A at (620,619) which is closest to patch 67 → depth range [0.251, 0.349]
        - B at (674,338) which is closest to patch 37 → depth range [0.0745, 0.1451]
        - C at (574,554) which is closest to patch 56 → depth range [0.1529, 0.251]
        Observation: A's patch has the highest maximum depth (0.349 > 0.251 > 0.145).
        Answer: (A)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """

    icl_4p_example = """
        You are given an image with multiple points circled on, and a <perceptionprogram> block. Using the image and <perceptionprogram> decide which point is the closest to the camera?

        One demonstrative example of how to use perception program to solve a certain problem is given below.

        Images: (scene with four labeled points A, B, C, D)
        Perception (depth in [0,1], higher = closer):
        - A at (212,603) → patch 51 → depth range [0.1686, 0.9569]
        - B at (471,322) → patch 37 → depth range [0.07059, 0.1333]
        - C at (90,484)  → patch 41 → depth range [0.07059, 0.9216]
        - D at (336,549) → patch 54 → depth range [0.1294, 0.2784]

        Observation: A's patch has the highest maximum depth (0.9569), making it closer than C, B, or D.
        Answer: (A)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """

    icl_5p_example = """
        You are given an image with multiple points circled on, and a <perceptionprogram> block. Using the image and <perceptionprogram> decide which point is the closest to the camera?

        One demonstrative example of how to use perception program to solve a certain problem is given below.

        Images: (scene with five labeled points A, B, C, D, E)
        Perception (depth in [0,1], higher = closer):
        - A at (588,568) → patch 56 → depth range [0.1608, 0.2667]
        - B at (28,500)  → patch 51 → depth range [0.1804, 0.9176]
        - C at (531,468) → patch 46 → depth range [0.1137, 0.1765]
        - D at (93,335)  → patch 31 → depth range [0.05882, 0.8745]
        - E at (831,625) → patch 69 → depth range [0.298, 0.3725]
        
        Observation: B's patch has the highest maximum depth (0.9176), greater than D (0.8745) and the others.
        Answer: (B)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """
    
    if k == 3:
        icl_example = icl_3p_example
    elif k == 4:
        icl_example = icl_4p_example
    else:
        icl_example = icl_5p_example

    system = '\n'.join([system_, icl_example])
    img = Image.open(sample["image"])
    question_ = sample["question"]
    prompt = '\n'.join([question_, depth_pp])

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt.strip()},
            ],
        },
    ]

def make_batches(idxs, n):
    for i in range(0, len(idxs), n):
        yield idxs[i:i+n]

@torch.inference_mode()
def generate_batch(conversations, max_new_tokens):
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs

def get_batched_perception_programs(idxs, coords_lookup, dataset):
    batched_pps = []
    for i in idxs:
        filename = os.path.basename(dataset[i]["image"])  # e.g., "0.png"
        assert filename in coords_lookup
        coords = coords_lookup[filename]
        coords_pp = coords_to_perceptionprogram(coords)
        depth_im_path = re.sub(
            r"(/blink\d+pointscenter/)images/(.+?)\.png$",
            r"\1depth/\2_depth.png",
            dataset[i]["image"]
        ).replace("_depth", "")
        depth_img = cv2.imread(depth_im_path, cv2.IMREAD_GRAYSCALE)
        depth_norm = depth_img.astype(np.float32) / 255.0
        depth_pp, _ = pp.emit_perception_program(
            modality="depth",
            field=depth_norm,
            seg=None,
            class_names=None,
            grid=(10, 10),
            add_relations=False,
            tau=0.15,
            relation_cap=0
        )
        to_append_depth_pp = "\n".join([coords_pp, depth_pp])
        batched_pps.append(to_append_depth_pp)
    return batched_pps

def run_qwen_pp_batched(output_root, data,
                        batch_size=BATCH_SIZE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        max_tries=MAX_TRIES,
                        retry_factor=RETRY_NEW_TOKENS_FACTOR,
                        max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    for k in data.keys():
        print(f"Processing HardBLINK-{k}!")
        dataset = data[k] # either 3, 4, 5
        out_path = os.path.join(output_root, f"hardblink_{k}_qwen3_4B_{mode.lower()}.jsonl")
        jsonl_path = f"/data00/hugo/hardblink/blink{k}pointscenter/point_positions.jsonl"
        coords_lookup = load_point_positions(jsonl_path)
        N = len(dataset)
        all_indices = list(range(N))
        results = {i: None for i in all_indices}
        
        attempt = 1
        pending = all_indices[:]
        wrong_counter = 0
        total_time_st = time.time()

        while pending and attempt <= max_tries:
            budget = int(min(max_new_tokens * (retry_factor ** (attempt - 1)), max_tokens_cap))
            pbar = tqdm(list(make_batches(pending, batch_size)),
                        desc=f"At: {attempt}/{max_tries} | Toks={budget}",
                        leave=False, dynamic_ncols=True)
            next_pending = []
            for idxs in pbar:
                samples = [dataset[i] for i in idxs]
                pps = get_batched_perception_programs(idxs, coords_lookup, dataset)
                conversations = [build_conversation(samples[j], k, pps[j]) for j in range(len(samples))]
                t0 = time.time()
                outputs = generate_batch(conversations, max_new_tokens=budget)
                t1 = time.time()

                for ds_idx, s, o in zip(idxs, samples, outputs):
                    pred = extract_choice(o)
                    rec = {
                        "image": s["idx"],
                        "question": s["question"],
                        "ground_truth": s["answer"],
                        "prediction": pred,
                        "output": o,
                        "correct": (pred == s["answer"]) if pred else False,
                        "attempt": attempt,
                        "max_new_tokens": budget,
                    }
                    results[ds_idx] = rec

                    if pred == "":
                        next_pending.append(ds_idx)
                    else:
                        wrong_counter += (not rec["correct"])

                pbar.set_postfix({"Pending": len(next_pending),
                                "WrongSoFar": wrong_counter,
                                "BatchTime": f"{t1 - t0:.2f}s"},
                                refresh=True)

            pending = next_pending
            attempt += 1

        total_time_end = time.time()

        final_list = [results[i] for i in sorted(results.keys()) if results[i] is not None]
        correct = sum(r["correct"] for r in final_list)
        acc = correct / len(final_list) if final_list else 0.0
        print(f"Final Accuracy: {acc:.4f} ({correct}/{len(final_list)})")
        print(f"Unanswered after retries: {len([r for r in final_list if r['prediction']==''])}")
        print(f"Total Time Taken: {total_time_end-total_time_st}s")
        
        with open(out_path, "w") as f:
            for r in final_list:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(final_list)} results → {out_path}")

def fetch_hardblink_data():
    all_n_choices = [3,4,5]
    data = {}
    root = "/data00/hugo/hardblink/"
    for n_choices in all_n_choices:
        task = f"blink{n_choices}pointscenter"
        data[n_choices] = []
        with jsonlines.open(os.path.join(root, task, f"blink{n_choices}pointscenter.jsonl")) as reader:
            for obj in reader:
                obj["idx"] = f"val_hardblink{n_choices}_{obj['image'].split('.')[0]}"
                obj["image"] = os.path.join(root, task, "images", obj["image"])
                obj["answer"] = f"({obj['answer']})"
                data[n_choices].append(obj)
    return data

if __name__ == "__main__":
    output_root = f"/data00/depth/open-source-models/qwen4b-{mode.lower()}/pp/"
    dataset = fetch_hardblink_data()
    run_qwen_pp_batched(output_root, dataset)