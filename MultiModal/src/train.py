import platform
import torch
import transformers
import argparse
from transformers import BlipForQuestionAnswering,BlipProcessor
MODEL_NAME = "Salesforce/blip-vqa-base"


# This project evaluates a pretrained BLIP model without fine-tuning because
# the collected dataset is an evaluation set rather than a training corpus.
# This script verifies the environment and model checkpoint used by the study.


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default = "blip",
        choices = ["blip"],
        help = "Pretrained VQA model to check",
    )

    parser.add_argument(
        "--device",
        default = "auto",
        choices = ["auto","cuda","cpu"],
        help = "Device used to load model"
    )

    return parser.parse_args()


def main():
    args = args_parse()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Python version:",platform.python_version())
    print("PyTorch version:",torch.__version__)
    print("Transformers version:",transformers.__version__)

    cuda_available = torch.cuda.is_available()
    print("CUDA available:",cuda_available)

    if cuda_available:
        print("CUDA device count:",torch.cuda.device_count())
        print("CUDA device name:",torch.cuda.get_device_name(0))

    try:
        processor = BlipProcessor.from_pretrained(MODEL_NAME)

        model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()
    except Exception as error:
        print("Model loading failed")
        print("Error:",error)
        raise SystemExit(1)

    model_device = next(model.parameters()).device

    print("Model name:",MODEL_NAME)
    print("Model device:",model_device)
    print("Environment check passed")


if __name__ == "__main__":
    main()
