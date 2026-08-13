from datasets import load_dataset
import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

mode = "Thinking" # "Instruct"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
dataset_size = len(dataset)
root = "/data00/visual_correspondence/"

model_name = f"Qwen/Qwen3-VL-4B-{mode}"
# keep this same
target_dir = "/data00/multiview_reasoning/open-source-models/"
processor = AutoProcessor.from_pretrained(f"Qwen/Qwen3-VL-4B-{mode}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=target_dir)

def call_qwen3_4b_vanilla(question):
    system =  \
    """
        You are an oracle answering multiple-choice questions about images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """

    img1 = question["image_1"]
    img2 = question['image_2']
    prompt = question['prompt'] + "\n" + question["question"]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
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
    generated_ids = model.generate(**inputs, max_new_tokens=32768)
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

def run_vanilla_qwen(output_root):
    pbar = tqdm(dataset, desc=f"Processing VisCorr!")
    this_output_file = os.path.join(output_root, f"qwen3_4B_{mode.lower()}_run02.jsonl")
    existing = {}
    wrong_counter = 0
    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        st = time.time()
        output = call_qwen3_4b_vanilla(sample)
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
        # count how many are wrong so far
        if not result['correct']:
            wrong_counter += 1
        pbar.set_postfix({
            'GT': sample['answer'],
            'Pred': prediction[:15],
            'Wrong': wrong_counter,
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
    output_root = f"/data00/visual_correspondence/open-source-models/qwen4b-{mode.lower()}/vanilla/"
    os.makedirs(output_root, exist_ok=True)
    run_vanilla_qwen(output_root)