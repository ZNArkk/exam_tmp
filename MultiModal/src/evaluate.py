import argparse
import json
from utils import (
    ensure_dir,
    is_correct,
    load_image,
    load_jsonl,
    resolve_project_path,
)
import torch
from transformers import BlipForQuestionAnswering,BlipProcessor
import csv
from pathlib import Path
MODEL_NAME = "Salesforce/blip-vqa-base"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default = str(PROJECT_DIR / "data" / "vqa.jsonl"),
        help = "Path to the JSONL dataset",
    )

    parser.add_argument(
        "--output",
        default = str(PROJECT_DIR / "results" / "answers.csv"),
        help = "Path to the output CSV file",
    )

    parser.add_argument(
        "--device",
        default = "auto",
        choices = ["auto","cuda","cpu"],
        help = "Device used for inference",
    )

    parser.add_argument(
        "--max_new_tokens",
        type = int,
        default = 20,
        help = "Maximum number of generated tokens"
    )

    return parser.parse_args()


def accuracy_by_type(results):
    type_total = {}
    type_correct = {}

    for result in results:
        question_type = result["type"]

        type_total[question_type] = (
            type_total.get(question_type,0) + 1
        )

        if result["is_correct"]:
            type_correct[question_type] = (
                type_correct.get(question_type,0) + 1
            )

    accuracy_by_type = {}

    for question_type,total in type_total.items():
        correct = type_correct.get(question_type,0)
        accuracy = correct / total
        accuracy_by_type[question_type] = accuracy

    print("Accuracy by type:")
    for question_type,accuracy in accuracy_by_type.items():
        print(f"{question_type}:{accuracy:.2%}")

    print()

    return accuracy_by_type


def accuracy_by_total(results):
    num_samples = len(results)

    num_correct = sum(
        1
        for result in results
        if result["is_correct"]
    )

    if num_samples > 0:
        overall_accuracy = num_correct / num_samples
    else:
        overall_accuracy = 0.0

    print("Number of samples:", num_samples)
    print("Number of correct answers:", num_correct)
    print(f"Overall accuracy:{overall_accuracy:.2%}")

    return num_samples,overall_accuracy


def model_run(args,device,processor,model,records):
    results = []

    for record in records:
        image_path = record["image"]
        question = record["question"]
        gt_answers = record["answers"]
        question_type = record["type"]

        image = load_image(image_path)

        inputs = processor(
            images = image,
            text = question,
            return_tensors = "pt",
        )

        inputs = {
            name : tensor.to(device)
            for name,tensor in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = args.max_new_tokens,
            )

        pred_answer = processor.decode(
            output_ids[0],
            skip_special_tokens = True,
        )

        correct = is_correct(pred_answer,gt_answers)

        result = {
            "image":image_path,
            "question":question,
            "gt_answers":gt_answers,
            "pred_answer":pred_answer,
            "type":question_type,
            "is_correct":correct,
        }

        results.append(result)

        print("Question:",question)
        print("Ground truth:",gt_answers)
        print("prediction",pred_answer)
        print("Correct:",correct)
        print()

    return results


def output_to_csv(args,results):
    output_path = resolve_project_path(args.output)

    ensure_dir(output_path.parent)

    fieldnames = [
        "image",
        "question",
        "gt_answers",
        "pred_answer",
        "type",
        "is_correct",
    ]

    with output_path.open(
        "w",
        encoding = "utf-8",
        newline = "",
    )as file:
        writer = csv.DictWriter(
            file,
            fieldnames = fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("Results saved to:",output_path)


def output_metrics(args,num_samples,overall_accuracy,type_accuracy):
    output_path = resolve_project_path(args.output)
    metrics_path = output_path.parent / "metrics.json"

    ensure_dir(metrics_path.parent)

    metrics = {
        "model" : "blip",
        "num_samples" : num_samples,
        "overall_accuracy" : overall_accuracy,
        "accuracy_by_type" : type_accuracy,
    }

    with metrics_path.open(
        "w",
        encoding = "utf-8",
    )as file:
        json.dump(
            metrics,
            file,
            ensure_ascii = False,
            indent = 2,
        )

    print("Metrics saved to:",metrics_path)


def main():
    args = parse_args()

    records = load_jsonl(args.data)

    print("Dataset:",args.data)
    print("Number of samples:",len(records))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Loading BLIP model...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    results = model_run(args,device,processor,model,records)

    type_accuracy = accuracy_by_type(results)
    num_samples,overall_accuracy = (
        accuracy_by_total(results)
    )

    output_to_csv(args,results)
    output_metrics(args,num_samples,overall_accuracy,type_accuracy)


if __name__ == "__main__":
    main()
