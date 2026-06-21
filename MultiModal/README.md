# MultiModal VQA Experiment

This project evaluates the pretrained `Salesforce/blip-vqa-base` model on a
small Visual Question Answering dataset collected for the assessment. The
submission contains the source code, 50 image-question samples, source
attribution, generated predictions, metrics, and analysis charts.

## Project Structure

```text
MultiModal/
|-- data/
|   |-- real_images/          # 50 experiment images and source metadata
|   |-- sources.md            # Dataset source notes
|   `-- vqa.jsonl             # Questions, accepted answers, and types
|-- results/
|   |-- answers.csv
|   |-- metrics.json
|   |-- accuracy_by_type.png
|   `-- error_cases.png
|-- src/
|   |-- train.py              # Environment and pretrained-model check
|   |-- evaluate.py           # Batch VQA evaluation
|   |-- inference.py          # Single-image inference
|   |-- analyze.py            # Metrics and error visualization
|   `-- utils.py              # Shared data and path utilities
|-- README.md
`-- requirements.txt
```

## Environment

The experiment was run with:

- Python 3.13.9
- PyTorch 2.11.0.dev20260103+cu128
- CUDA 12.8
- Transformers 5.12.0

Install the dependencies from the repository root:

```bash
pip install -r MultiModal/requirements.txt
```

The BLIP checkpoint is downloaded automatically from Hugging Face on the first
run. A GPU is recommended but not required.

## Dataset

`data/vqa.jsonl` contains 50 samples across object, color, count, action,
scene, and OCR-style question types. Each line is a JSON object with the
following fields:

```json
{"image": "MultiModal/data/real_images/object/object_01_dog.jpg", "question": "What animal is shown in the image?", "answers": ["dog", "a dog"], "type": "object"}
```

Image attribution and metadata are retained in `data/sources.md` and
`data/real_images/`.

## Reproduce the Experiment

Run all commands from the cloned assessment repository root.

1. Check the environment and pretrained model:

```bash
python MultiModal/src/train.py --device auto
```

The experiment uses BLIP without fine-tuning because the collected samples are
an evaluation dataset, not a training corpus. Therefore, `train.py` verifies
the environment and model checkpoint instead of updating model weights.

2. Evaluate all samples:

```bash
python MultiModal/src/evaluate.py --device cuda
```

Use `--device cpu` on a machine without CUDA. The default input is
`MultiModal/data/vqa.jsonl`, and the default output is
`MultiModal/results/answers.csv`.

3. Generate analysis charts:

```bash
python MultiModal/src/analyze.py
```

4. Run one question:

```bash
python MultiModal/src/inference.py --image MultiModal/data/real_images/object/object_01_dog.jpg --question "What animal is shown in the image?" --device auto
```

## Results

The 50-sample experiment achieved an overall exact-match accuracy of **84%**.
Accuracy by type was:

| Type | Accuracy |
| --- | ---: |
| Object | 100% |
| Color | 100% |
| Count | 80% |
| Action | 40% |
| Scene | 60% |
| OCR | 90% |

The detailed predictions are stored in `results/answers.csv`. The main failure
cases involve fine-grained counting, action recognition, and ambiguous scene
descriptions. OCR-style samples are retained as limitation-analysis cases and
do not add a separate OCR branch to the VQA pipeline.

## Model and Tool Attribution

- BLIP: `Salesforce/blip-vqa-base`
- Transformers: https://github.com/huggingface/transformers
- Model paper: Li et al., "BLIP: Bootstrapping Language-Image Pre-training for
  Unified Vision-Language Understanding and Generation," ICML 2022.

AI tools were used for coding guidance, debugging explanations, and document
formatting. The experiment execution, dataset selection, result inspection,
and final submission review were completed by the author.
