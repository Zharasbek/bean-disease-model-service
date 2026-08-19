from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


CHECKPOINT = "microsoft/resnet-18"
OUTPUT_DIR = Path("artifacts/resnet18-beans")
FINAL_MODEL_DIR = Path("model")
REPORTS_DIR = Path("reports")
SEED = 42


def main():
    set_seed(SEED)

    # 1. Download the dataset
    dataset = load_dataset("AI-Lab-Makerere/beans")

    # 2. Extract class names
    label_names = dataset["train"].features["labels"].names
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in enumerate(label_names)}

    print("Classes:", label_names)

    # 3. Download the preprocessing configuration
    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

    if "shortest_edge" in image_processor.size:
        image_size = image_processor.size["shortest_edge"]
    else:
        image_size = (
            image_processor.size["height"],
            image_processor.size["width"],
        )

    normalize = Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )

    # Random operations are used only during training.
    train_augmentations = Compose(
        [
            RandomResizedCrop(image_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    # Validation and test preprocessing must be deterministic.
    evaluation_transforms = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            normalize,
        ]
    )

    def transform_training_batch(examples):
        examples["pixel_values"] = [
            train_augmentations(image.convert("RGB"))
            for image in examples["image"]
        ]
        del examples["image"]
        return examples

    def transform_evaluation_batch(examples):
        examples["pixel_values"] = [
            evaluation_transforms(image.convert("RGB"))
            for image in examples["image"]
        ]
        del examples["image"]
        return examples

    training_dataset = dataset["train"].with_transform(
        transform_training_batch
    )
    validation_dataset = dataset["validation"].with_transform(
        transform_evaluation_batch
    )
    test_dataset = dataset["test"].with_transform(
        transform_evaluation_batch
    )

    # 4. Download ResNet-18 and replace its classifier.
    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # The warning about newly initialized classifier weights is expected.

    def compute_metrics(evaluation_prediction):
        logits = evaluation_prediction.predictions
        true_labels = evaluation_prediction.label_ids
        predicted_labels = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "macro_f1": f1_score(
                true_labels,
                predicted_labels,
                average="macro",
            ),
        }

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        remove_unused_columns=False,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,

        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=0.1,

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,

        num_train_epochs=10,
        logging_steps=10,
        save_total_limit=2,

        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # Safest setting for your previous Windows memory problem.
        dataloader_num_workers=0,

        # Establish a stable baseline before experimenting with FP16.
        fp16=False,

        report_to="none",
        seed=SEED,
        data_seed=SEED,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    # 5. Fine-tune
    trainer.train()

    # 6. Evaluate once on the untouched test split
    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    print("\nTest metrics:")
    print(test_metrics)

    # 7. Generate detailed test predictions
    prediction_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(
        prediction_output.predictions,
        axis=1,
    )
    true_labels = prediction_output.label_ids

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=label_names,
            digits=4,
        )
    )

    # 8. Save the final model and processor
    FINAL_MODEL_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    image_processor.save_pretrained(str(FINAL_MODEL_DIR))

    # 9. Save confusion matrix
    REPORTS_DIR.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        display_labels=label_names,
        cmap="Blues",
        xticks_rotation=30,
    )

    plt.title("ResNet-18 Beans Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    print(f"\nModel saved to: {FINAL_MODEL_DIR.resolve()}")
    print(f"Reports saved to: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
