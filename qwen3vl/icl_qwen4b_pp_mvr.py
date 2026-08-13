from datasets import load_dataset
import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import generate_perception_program as pp
import numpy as np

dataset = load_dataset("BLINK-Benchmark/BLINK", "Multi-view_Reasoning",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
dataset_size = len(dataset)
root = "/data00/multiview_reasoning/"

model_name = "Qwen/Qwen3-VL-4B-Thinking"
target_dir = "/data00/multiview_reasoning/open-source-models/"
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=target_dir)

def call_qwen3_4b_with_pp(question, flow_pp):
    system = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. When answering, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    
    icl_example = """
        You're given two frames from a static scene and a <perceptionprogram> block that lists per-patch horizontal motion
        ("left" or "right") on a 10x10 grid. Decide the global camera motion:
        (A) left    (B) right
        
        One example of how to use perception program to solve the problem is given below.

        Images: (beginning frame, end frame)
        Perception: many 'right' patches in upper rows; some 'left' in lower rows
        Justification: The upper rows overwhelmingly show 'right' motion and overall the grid has a right majority (roughly 60+ vs 40), indicating global camera motion to the right.
        Answer: (B)

        Now use this and solve the given question.
    """

    img1 = question["image_1"]
    img2 = question['image_2']
    question_ = question['prompt'] + "\n" + question["question"]
    prompt = '\n'.join([question_, flow_pp])

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
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return output_text

def extract_choice(text: str):
    matches = re.findall(r'(?<!\w)\(\s*([ABab])\s*\)(?!\w)', text)
    return f"({matches[-1].upper()})" if matches else None

def run_qwen_with_pp(output_root):
    pbar = tqdm(dataset, desc=f"Processing MVR!")
    this_output_file = os.path.join(output_root, f"qwen3_4B_thinking_with_pp.jsonl")
    flow_path = "/data00/multiview_reasoning/flow_data/"
    existing = {}
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

        st = time.time()
        output = call_qwen3_4b_with_pp(sample, flow_pp)
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
    output_root = "/data00/multiview_reasoning/open-source-models/qwen4b-thinking/pp/"
    os.makedirs(output_root, exist_ok=True)
    run_qwen_with_pp(output_root)