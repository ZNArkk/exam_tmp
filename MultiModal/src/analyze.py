import matplotlib
matplotlib.use("Agg")
import pandas as pd
import argparse
from utils import ensure_dir, load_image, resolve_project_path
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default = str(PROJECT_DIR / "results" / "answers.csv"),
        help = "Path to the evaluation CSV file"
    )

    parser.add_argument(
        "--out_dir",
        default = str(PROJECT_DIR / "results"),
        help = "Directory used to save charts"
    )

    return parser.parse_args()


def get_chart(args,accuracy_to_type):
    out_dir = ensure_dir(args.out_dir)
    chart_path = out_dir/"accuracy_by_type.png"

    fig,ax = plt.subplots(figsize = (8,5))

    ax.bar(
        accuracy_to_type.index,
        accuracy_to_type.values,
        color = "steelblue",
    )

    ax.set_title("VQA Accuracy by Question Type")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1.05)

    fig.tight_layout()
    fig.savefig(chart_path,dpi = 150)

    plt.close(fig)

    print("Accuracy saved to:",chart_path)


def get_error_cases(dataframe,args):
    error_cases = dataframe[
        dataframe["is_correct"] == False
    ]
    error_cases = error_cases.head(6)

    out_dir = ensure_dir(args.out_dir)
    error_chart_path = out_dir/"error_cases.png"

    if error_cases.empty:
        fig,ax = plt.subplots(figsize = (8,4))

        ax.text(
            0.5,
            0.5,
            "No error cases found",
            ha = "center",
            va = "center",
        )

        ax.axis("off")

    else:
        num_cases = len(error_cases)
        num_cols = min(3,num_cases)
        num_rows = (num_cases + num_cols - 1) // num_cols

        fig,axes = plt.subplots(
            num_rows,
            num_cols,
            figsize = (5 * num_cols,4 * num_rows),
            squeeze = False,
        )

        for ax,(_,row) in zip(axes.flat,error_cases.iterrows()):
            image = load_image(row["image"])

            ax.imshow(image)
            ax.axis("off")

            title = (
                f"Q:{row['question']}\n"
                f"GT:{row['gt_answers']}\n"
                f"Pred:{row['pred_answer']}\n"
                f"Type:{row['type']}"
            )

            ax.set_title(title,fontsize = 9)

        for ax in list(axes.flat)[num_cases:]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(error_chart_path,dpi = 150)
        plt.close(fig)

    print("Number of error cases:",len(error_cases))
    for _,row in error_cases.iterrows():
        print("Question:",row["question"])
        print("Ground truth:",row["gt_answers"])
        print("Prediction:",row["pred_answer"])
        print("Type:",row["type"])
        print()
    print("Error chart saved to:",error_chart_path)

def main():
    args = parse_args()

    dataframe = pd.read_csv(resolve_project_path(args.input))

    print("Number of results:",len(dataframe))
    print(dataframe.head())

    overall_accuracy = dataframe["is_correct"].mean()
    accuracy_by_type = dataframe.groupby("type")["is_correct"].mean()

    print(f"Overall accuracy:{overall_accuracy:.2%}")
    print("Accuracy by type:")
    for question_type,accuracy in accuracy_by_type.items():
        print(f"{question_type}:{accuracy:.2%}")

    get_chart(args,accuracy_by_type)
    get_error_cases(dataframe,args)


if __name__ == "__main__":
    main()
