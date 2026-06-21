import argparse
from utils import load_image
import torch
from transformers import BlipForQuestionAnswering,BlipProcessor
MODEL_NAME = "Salesforce/blip-vqa-base"


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run VQA inference on one image"
    )

    parser.add_argument(
        "--image",
        required = True,
        help = "Path to the input image",
    )

    parser.add_argument(
        "--question",
        required = True,
        help = "Question about the image",
    )

    parser.add_argument(
        "--device",
        default = "auto",
        choices = ["auto","cuda","cpu"],
        help = "Device : auto ,cpu or cuda",
    )

    parser.add_argument(
        "--max_new_tokens",
        type = int,
        default = 20,
        help = "Maximum number of tokens generated for the answer",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    image = load_image(args.image)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Loading BLIP model...")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    inputs = processor(
        images = image,
        text = args.question,
        return_tensors = "pt",
    )

    inputs = {
        name : tensor.to(device)
        for name,tensor in inputs.items()
    }

    for name,tensor in inputs.items():
        print(name,tensor.shape,tensor.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
        )

    answer = processor.decode(
        output_ids[0],
        skip_special_tokens = True,
    )

    print("Question:",args.question)
    print("Device:",device)
    print("Answer:",answer)

if __name__ == "__main__":
    main()
