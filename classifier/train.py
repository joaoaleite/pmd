# train.py

import argparse
import os
import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import (
    SemEvalClassifier,
    SEMEVAL_LABELS,
    id2label,
    label2id,
    load_train_dev_semeval,
    get_dataloader,
)

parser = argparse.ArgumentParser(description="Train a sequence classification model.")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=3.4e-5,
    help="Learning rate for the optimizer.",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training and validation."
)
parser.add_argument(
    "--num_epochs", type=int, default=10, help="Number of training epochs."
)
parser.add_argument(
    "--logging_steps", type=int, default=100, help="Steps interval for logging."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="roberta-large",
    help="Model name or path to pretrained model.",
)
parser.add_argument(
    "--clf_threshold", type=float, default=0.2, help="Threshold for classification."
)

# Load environment variables
load_dotenv()

cuda_order = os.getenv("CUDA_DEVICE_ORDER")
cuda_device = os.getenv("CUDA_DEVICE_NUM")

# Set CUDA device
os.environ["CUDA_DEVICE_ORDER"], os.environ["CUDA_VISIBLE_DEVICES"] = (
    cuda_order,
    cuda_device,
)
print(f"Using CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")


if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize W&B project
    wandb.init(
        project="persuasion-multi-domain",
        config={
            "model_name": args.model_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
        },
    )

    # Initialize the model and trainer
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(SEMEVAL_LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SemEvalClassifier(
        model=model, learning_rate=args.learning_rate, clf_threshold=args.clf_threshold
    )

    # Load and prepare data
    df_train, df_dev = load_train_dev_semeval()

    # Create datasets and data loaders
    train_loader = get_dataloader(df_train, tokenizer, args.batch_size)
    dev_loader = get_dataloader(df_dev, tokenizer, args.batch_size)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=args.logging_steps,
        val_check_interval=0.25,
        accelerator="cuda",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="eval_f1_micro",
                mode="max",
                dirpath="classifier/checkpoints",
                filename=args.model_name + "_{eval_f1_micro:.2f}",
                auto_insert_metric_name=False,
            )
        ],
        logger=pl.loggers.WandbLogger(project="persuasion-multi-domain"),
    )

    # Train and evaluate
    trainer.fit(model, train_loader, dev_loader)
    wandb.finish()
