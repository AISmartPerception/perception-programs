from datasets import load_dataset
import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

mode = "Thinking"
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

def call_qwen3_4b_tool(question, loftr_img):
    system = """
        You are an oracle answering multiple-choice questions about images. You will be provided two images (image 1 and image 2) and their corresponding visual correspondence map, you should use both to answer. In the visual correspondence map (left image is image 1, right image is image 2), each line connects two similar points in the two images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    icl_example = """
        You're given two images of the same scene and a visual correspondence map (colored tracks/lines
        that link matching points between the two frames). Your job: choose which red-circled point in
        IMAGE 2 (A/B/C/D) corresponds to the single reference point (REF) in IMAGE 1.

        One example of how to use the visual correspondence map to solve the task:

        Images: (reference frame with REF, target frame with A/B/C/D)
        Correspondence map: the tracks that originate at REF (on the construction joint near the bottom-center)
        run horizontally to the right and terminate at the same joint region in image 2.
        Justification: The lines anchored at REF converge near candidate C; tracks near A head to the roof area,
        near B to the statue body, and near D into the trees—none align with the REF region.
        Answer: (C)

        Now use this idea and solve the given question.
    """
    
    img1 = question["image_1"]
    img2 = question['image_2']
    loftr_map = Image.open(loftr_img)
    prompt = question['prompt'] + "\n" + question["question"]
    system_prompt = '\n'.join([system, icl_example])
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": "Visual Correspondence Map"},
                {"type": "image", "image": loftr_map},
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
    # by default, do_sample=False, and num_beams=1, so greedy decoding is used.
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

def run_qwen_with_tool(output_root):
    pbar = tqdm(dataset, desc=f"Processing VisCorr!")
    this_output_file = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")
    existing = {}
    loftr_imgs = "/data00/visual_correspondence/tool_output/"
    wrong_counter = 0
    for i, sample in enumerate(pbar):
        img_id = sample['idx']
        loftr_map_path = loftr_imgs + img_id + ".png"
        st = time.time()
        output = call_qwen3_4b_tool(sample, loftr_map_path)
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
    output_root = f"/data00/visual_correspondence/open-source-models/qwen4b-{mode.lower()}/tool/"
    os.makedirs(output_root, exist_ok=True)
    run_qwen_with_tool(output_root)