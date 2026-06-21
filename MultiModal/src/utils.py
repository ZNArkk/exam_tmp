from pathlib import Path
from PIL import Image
import re
import string
import json


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_project_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    if path.exists():
        return path.resolve()

    if path.parts and path.parts[0].lower() == "multimodal":
        path = Path(*path.parts[1:])

    return PROJECT_DIR / path


def load_image(image_path):
    path = resolve_project_path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    image = image.convert("RGB")
    return image


def normalize_answer(text):
    text = str(text).lower().strip()

    for p in string.punctuation:
        text = text.replace(p," ")

    text = re.sub(r"\s+"," ",text).strip()

    number_map = {
        "one" : "1",
        "two" : "2",
        "three" : "3",
        "four" : "4",
        "five" : "5",
        "six" : "6",
        "seven" : "7",
        "eight" : "8",
        "nine" : "9",
    }

    words = []
    for word in text.split():
        if word in number_map:
            words.append(number_map[word])
        else:
            words.append(word)

    return " ".join(words)


def is_correct(pred,gt_answers):
    pred = normalize_answer(pred)

    if isinstance(gt_answers,str):
        gt_answers = [gt_answers]

    for answer in gt_answers:
        answer = normalize_answer(answer)
        if pred == answer:
            return True

    return False


def load_jsonl(path):
    path = resolve_project_path(path)
    records = []

    with path.open("r",encoding = "utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records.append(record)

    return records


def ensure_dir(path):
    path = resolve_project_path(path)
    path.mkdir(parents = True,exist_ok = True)
    return path
