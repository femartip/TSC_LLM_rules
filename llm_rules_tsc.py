import argparse
import io
import os
import re
import numpy as np
import base64
from dotenv import load_dotenv, dotenv_values
from Utils.load_models import model_batch_classify
from Utils.selectPrototypes import select_prototypes, select_random_timeseries
from Utils.load_data import load_dataset, load_dataset_labels, normalize_data
from Utils.save_llm_results import save_raw_outputs_txt, save_results
import openai
from openai import OpenAI
import matplotlib
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
import logging
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

load_dotenv()

api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY not found, add it to .env file")

client = OpenAI(api_key=api_key)

DEBUG = False
logger = logging.getLogger(__name__)

def should_retry_openai_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True

    if isinstance(exc, openai.BadRequestError):
        error_message = str(exc).lower()
        if "could not parse the json body of your request" in error_message:
            return True

    return False

@retry(
    retry=retry_if_exception(should_retry_openai_error),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_response(prompt: list[dict], model:str, reasoning_effort: str = "high") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], #type: ignore
        #temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    return response.choices[0].message.content


### baseline experiment no rules
def build_baseline_prompt(images: list[str], test_samples: list[str], num_labels: int) -> list[dict]:
    labels = range(0, num_labels)
    classes = ", ".join(map(str, labels))
    num_images_per_label = len(images) // num_labels
    
    prompt = [
        {"type": "text", "text": f"""You are a time-series classification expert. 
         Your goal is to learn from labeled examples (classes {classes}) and classify the new instances.
         
         1. Examine the labeled examples to identify class-specific patterns.
         2. Compare the new, unlabeled instances to the characteristics of classes {classes}.
         3. Provide a brief rationale for your classification.
         4. Conclude with exactly {len(test_samples)} lines, one for each test instance, in the format: "Predicted class: <label>"."""}
    ]
    
    remaining_images = list(images)
    
    for label in labels:
        images_label = remaining_images[:num_images_per_label]
        remaining_images = remaining_images[num_images_per_label:]
        
        prompt.append({"type": "text", "text": f"Class {label} examples ({len(images_label)} time-series plots):"})
        prompt.extend([{"type": "image_url", "image_url": {"url": img}} for img in images_label]) #type: ignore

    prompt.append({"type": "text", "text": "New instances to classify (unlabeled):"})
    prompt.extend([{"type": "image_url", "image_url": {"url": ts}} for ts in test_samples]) #type: ignore
    
    return prompt

def prompt_baseline_model(llm_model: str, k_img: list[str], test_sample: list[str], test_labels: list[int], num_labels: int):
    # Build prompt
    prompt = build_baseline_prompt(k_img, test_sample, num_labels)
    response = get_response(prompt, llm_model, reasoning_effort="high")

    # Regex parse
    pattern = r"Predicted class:\s+(\d+)"
    preds = [int(x) for x in re.findall(pattern, response)]

    # Calculate accuracy
    count = min(len(test_labels), len(preds))
    acc = sum(1 for i in range(count) if preds[i] == test_labels[i]) / len(test_labels)

    return acc, preds, response


### PROMPT BUILDERS FOR RULE EXTRACTION AND CLASSIFICATION WITH RULES
def build_rule_prompt(images: list[str], num_labels: int, n_rules: int):
    labels = range(0, num_labels)
    num_images = len(images)    # num prototypes
    classes = ", ".join(map(str, labels))

    prompt = [
        {"type": "text",
        "text": f"""

        You are a time-series classification expert analyzing labeled prototypes for classes {classes} ({num_images} prototypes per class).
        Follow these steps:
        Step 1 — Analyze differences between classes Identify which regions (early, middle, late) differ most, and whether differences are best described by: thresholds, trends, peaks/troughs, plateaus, or temporal shape patterns.
        Step 2 — Build a feature summary Determine which of the following are most discriminative between classes:
            • Region statistics (mean/min/max in early, middle, late)
            • Trends (rising/falling)
            • Peaks, troughs, plateaus
            • Relative differences between regions
        Step 3 — Generate human-readable classification rules Each rule must:
            • Describe one main concept
            • Use either a numeric comparison or a descriptive shape term (e.g. upward peak, broad plateau, falling trend, rising tail)
            • Be as concise as possible
        Avoid: redundant rules, conditions shared by all classes, mathematical notation, ambiguity.
        Step 4 — Validate internally Check that the rules correctly distinguish the prototypes. Refine any non-discriminative rules. Prefer the smallest rule set that separates all classes.
        
        Output format (strictly):
        Class <label>:
        R1: ...
        R2: ...
        ...

        """}
        ]

    num_images_per_label = int(num_images / num_labels)

    for label in labels:
        images_label = images[:num_images_per_label]
        images = images[num_images_per_label:]

        prompt.append({"type": "text", "text": f"Class {label} examples:"})
        prompt.extend([{"type": "image_url", "image_url": {"url": img}} for img in images_label]) #type: ignore

    return prompt

def build_classification_prompt(rule: str, test_samples: list[str]):
    num_samples = len(test_samples)
    prompt = [
        {
        "type": "text",
        "text": f"""
        You are given the following decision rules:
        
        {rule}
        
        Apply this rule to classify the {num_samples} new time-series plots below.
        
        Instructions:
        - Follow the rule strictly.
        - Do not invent new criteria.
        - The labels you output must come only from the class labels given in the ruleset.
        - Do not add bullets, numbering.
        - Do not include explanations, uncertainty, or alternative class labels.
        - If uncertain, still choose the best class.
        - The output must contain exactly {num_samples} lines, in the same order as the images.
        - For each prediction, use the exact format: 'Predicted class: X'
        - X must be one of the allowed labels from the ruleset.

        Examples of valid lines:
        Predicted class: 0
        Predicted class: 1
        Predicted class: 2
        """}
        ]

    prompt.extend([{"type": "image_url", "image_url": {"url": img}} for img in test_samples])   #type: ignore

    return prompt


def get_idx_per_cls(labels_ts: np.ndarray, k_cls:int) -> dict[str,list[int]]:
    labels = np.unique(labels_ts)
    idx_labels = {}
    for label in labels:
        labels_idx = np.where(labels_ts == label)[0]
        rand_idx = np.asanyarray(np.random.randint(labels_idx.shape[0], size=(k_cls)))
        idx_labels[label] = rand_idx

    return idx_labels

def get_k_examples(dataset_ts: np.ndarray, k_idx:dict) -> np.ndarray:
    labels = k_idx.keys()
    k_examples = []
    for label in labels:
        idx = k_idx[label]
        k_examples_label = dataset_ts[idx]
        k_examples.append(k_examples_label)

    return np.array(k_examples)

def ts_to_image(ts: np.ndarray, show_fig: bool = False, name: str = ""):
    plt.figure(figsize=(4,3))
    plt.plot(ts); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf)
    if show_fig:
        plt.savefig(f"./llm_tests/{name}")
        plt.pause(1)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_b64}"

def simp_ts_to_img(dataset_ts: np.ndarray, dataset_ts_labels: list[int], test_ts: np.ndarray) -> tuple[list[str], list[str]]:
    dataset_ts = dataset_ts
    dataset_ts_labels = dataset_ts_labels
    test_ts = test_ts
    
    k_img = [ts_to_image(ts, show_fig=DEBUG, name=f"train_{i}") for i, ts in enumerate(dataset_ts)]
    test_sample = [ts_to_image(ts, show_fig=DEBUG, name=f"test_{i}") for i, ts in enumerate(test_ts)]

    return k_img, test_sample


### NEW FUNCTIONS FOR RULE EXTRACTION AND CLASSIFICATION WITH RULES
def extract_rule(llm_model: str, k_img: list[str], labels: int, n_rules: int):
    prompt = build_rule_prompt(k_img, labels, n_rules)
    rule = get_response(prompt, llm_model, reasoning_effort="high")   ### changes the reasoning effort for rule generation
    return rule

### batch classify
def batch_classify_with_rule(llm_model: str, rule: str, test_imgs: list[str], test_labels: list[int], batch_size: int = 10):
    all_predicted_labels = []
    raw_batch_responses = []
    
    # Process test_imgs in chunks
    for i in range(0, len(test_imgs), batch_size):
        chunk_imgs = test_imgs[i : i + batch_size]
        
        # Build prompt for just this chunk
        prompt = build_classification_prompt(rule, chunk_imgs)
        response = get_response(prompt, llm_model, reasoning_effort="high")    ### changes the reasoning effort for classification
        raw_batch_responses.append({
            "batch_start": i,
            "batch_size": len(chunk_imgs),
            "response": response,
        })
        
        # Parse predictions for this chunk
        pattern = r"Predicted class:\s+(\d+)"
        batch_preds = [int(x) for x in re.findall(pattern, response)]

        # check if LLM got all time series
        if len(batch_preds) != len(chunk_imgs):
            print(f"ALIGNMENT ERROR: Expected {len(chunk_imgs)} labels, got {len(batch_preds)}. Padding with -1.")
            diff = len(chunk_imgs) - len(batch_preds)
            batch_preds.extend([-1] * diff)
        
        # Handle cases where LLM might return fewer/more preds than images
        # This is a safe way to append what you got
        all_predicted_labels.extend(batch_preds)
        
    # Calculate accuracy across the full list
    count = min(len(test_labels), len(all_predicted_labels))
    accuracy = sum(1 for i in range(count) if all_predicted_labels[i] == test_labels[i]) / len(test_labels)
    
    return accuracy, all_predicted_labels, raw_batch_responses


### rule swap
def swap_rules_robust(rules_text: str) -> str:
    # 1. Regex to split the text into list of (Header, Content) tuples
    # Matches "Class X:" and the following text
    pattern = r"(Class \d+:)(.*?)(?=Class \d+:|$)"
    matches = re.findall(pattern, rules_text, re.DOTALL)
    
    if len(matches) < 2:
        return rules_text

    # 2. Extract content and labels
    headers = [m[0] for m in matches]
    contents = [m[1] for m in matches]
    
    # 3. Swap the contents, but keep the headers in their original order
    # This puts Class 0's content under Class 1's header, and vice-versa
    swapped_text = f"{headers[0]}{contents[1]}\n\n{headers[1]}{contents[0]}"
    
    return swapped_text

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="miniRocket", help="Classifier to compare with." )
    parser.add_argument('--llm', type=str, default="gpt-5.1",help="LLM within the OpenAI API. Models supported: gpt4o, o4-mini, gpt-4.1 and o3")
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--rules', type=int, default=2, help="Number of LLM generated classification rules")
    parser.add_argument('--mode', type=str, default="rulebased", choices=["rulebased", "baseline", "noPrototype", "baselineNoPrototype"], help="Mode of the experiment.")
    parser.add_argument('--save_raw_outputs', action='store_true', help="Save raw model text outputs in separate txt files.")
    parser.add_argument('--prompt_version', default="promptV2", help='Naming scheme to differentiate experiments, promptV1')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    assert args.llm in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "gpt-5.1", ], "Invalid model type"
    print(f"Testing {args.mode} using {args.dataset } for classifier {args.classifier} on LLM {args.llm} using {args.rules} LLM rules")

    rule = "Baseline (No rules)"
        
    full_train_ts_norm = load_dataset(args.dataset, data_type="TRAIN_normalized")
    full_test_ts_norm = load_dataset(args.dataset, data_type="TEST_normalized")

    for n in range(3):
        raw_outputs = None
        rand_ts_idx = np.random.randint(0, full_test_ts_norm.shape[0], size=(100))    # edit size=(n) to change how many to classify per run
        test_ts_norm = full_test_ts_norm[rand_ts_idx]

        if args.mode in ["rulebased", "baseline"]:
            prototipes_ts_norm, support_examples = select_prototypes(args.dataset, num_instances=args.k, data_type="TRAIN_normalized", return_metadata=True) # change metric to preferred distance measure (dtw, euclidean, etc), metric="euclidean"
        elif args.mode in ["noPrototype", "baselineNoPrototype"]:
            prototipes_ts_norm, support_examples = select_random_timeseries(args.dataset, num_instances=args.k, data_type="TRAIN_normalized", return_metadata=True)  

        prot_labels = np.array(load_dataset_labels(args.dataset, data_type='TEST_normalized'))
        classifier_file = f"{args.classifier}_norm.pth" if args.classifier == "cnn" else f"{args.classifier}_norm.pkl"
        dataset_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", prototipes_ts_norm, len(set(prot_labels)))   #type: ignore
        test_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", test_ts_norm, len(set(prot_labels)))  #type: ignore
        prot_img_simp, test_img_simp = simp_ts_to_img(prototipes_ts_norm, dataset_ts_labels, test_ts_norm)

        if args.mode in ["baseline", "baselineNoPrototype"]:
            accuracy, preds, baseline_response = prompt_baseline_model(args.llm, prot_img_simp, test_img_simp, test_ts_labels, len(set(prot_labels)))
            if args.save_raw_outputs:
                raw_outputs = {
                    "baseline_response": baseline_response,
                }
        
        elif args.mode in ["rulebased", "noPrototype"]:
            # generate ruleset
            rule = extract_rule(args.llm, prot_img_simp, len(set(prot_labels)), args.rules)
            print("Extracted Rule:\n", rule)
            
            # batch classifier
            accuracy, preds, raw_batch_responses = batch_classify_with_rule(
                args.llm, 
                rule, 
                test_img_simp, 
                test_ts_labels, 
                batch_size=10  # adjust size per batch
            )
            if args.save_raw_outputs:
                raw_outputs = {
                    "rule_generation_response": rule,
                    "classification_batch_responses": raw_batch_responses,
                }
        else:
            raise ValueError(f"Unknown mode: {args.mode}")    

        print("\n")
            
        # table
        print(f"{'Instance':<10} | {'True Label':<10} | {'Predicted':<10} | {'Status':<10} | {'TS Idx':<10}")
        print("-" * 70)
        for idx, (true_l, pred_l, ts_idx) in enumerate(zip(test_ts_labels, preds, rand_ts_idx)):
            status = "MATCH" if true_l == pred_l else "MISMATCH"
            print(f"{idx+1:<10} | {true_l:<10} | {pred_l:<10} | {status:<10} | {ts_idx:<10}")
            
        print("-" * 70)
        print(f"Final Accuracy: {accuracy * 100}%")

        # save results as json file in results as "llm_rules_results"
        save_results(args, rule, accuracy, preds, test_ts_labels, rand_ts_idx, repetition=n + 1, support_examples=support_examples)
        if args.save_raw_outputs and raw_outputs is not None:
            save_raw_outputs_txt(args, raw_outputs, repetition=n + 1)
