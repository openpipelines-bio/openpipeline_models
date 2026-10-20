import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
from datasets import load_from_disk
from geneformer import Classifier
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

## VIASH START
par = {
    "input": "test.dataset",
    "id_class_dict": "id_class_dict.pkl",
    "model": "finetuned_model",
    "model_version": "V1",
    "obs_report_by": None,
    "forward_batch_size": 32,
    "output_metrics": "metrics.csv",
    "output_confusion": "confusion.csv",
    "output_predictions": "predictions.csv",
}
meta = {"resources_dir": "src/utils", "cpus": 4, "temp_dir": None}
## VIASH END

sys.path.append(meta["resources_dir"])
from setup_logger import setup_logger  # noqa: E402

logger = setup_logger()


def metrics(cells, labels):
    """Accuracy, macro F1 over the labels present, per-label precision and
    recall and, for 2 labels, the ROC AUC of the second label, over the cells
    of `cells`."""
    y_true, y_pred = cells["true_label"], cells["predicted_label"]
    row = {
        "n_cells": len(cells),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for label, p, r in zip(labels, precision, recall):
        row[f"n_{label}"] = int((y_true == label).sum())
        row[f"precision_{label}"] = p
        row[f"recall_{label}"] = r
    # undefined when the cells hold one label only, e.g. for a single donor
    row["roc_auc"] = np.nan
    if len(labels) == 2 and y_true.nunique() == 2:
        row["roc_auc"] = roc_auc_score(y_true == labels[1], cells[f"prob_{labels[1]}"])
    return row


def main():
    report_by = par["obs_report_by"] or []
    data = load_from_disk(par["input"])
    absent = [
        column for column in ["cell_id", *report_by] if column not in data.column_names
    ]
    if absent:
        raise ValueError(
            f"Columns {absent} not found in --input. Available: {data.column_names}"
        )
    with open(par["id_class_dict"], "rb") as f:
        id_class_dict = pickle.load(f)
    labels = [id_class_dict[i] for i in sorted(id_class_dict)]

    classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "label", "states": "all"},
        forward_batch_size=par["forward_batch_size"],
        model_version=par["model_version"],
        nproc=int(meta["cpus"] or 1),
    )
    with tempfile.TemporaryDirectory(dir=meta.get("temp_dir")) as temp_dir:
        result = classifier.evaluate_saved_model(
            model_directory=par["model"],
            id_class_dict_file=par["id_class_dict"],
            test_data_file=par["input"],
            output_directory=temp_dir,
            output_prefix="test",
            predict=True,
            predict_metadata=["cell_id", *report_by],
        )
        with open(f"{temp_dir}/test_pred_dict.pkl", "rb") as f:
            pred_dict = pickle.load(f)
    logger.info(
        "Geneformer: macro F1 %.4f, accuracy %.4f", result["macro_f1"], result["acc"]
    )

    probabilities = softmax(np.asarray(pred_dict["predictions"], dtype=float), axis=1)
    predictions = pd.DataFrame(pred_dict["prediction_metadata"])
    predictions["true_label"] = [id_class_dict[i] for i in pred_dict["label_ids"]]
    predictions["predicted_label"] = [labels[i] for i in probabilities.argmax(axis=1)]
    for i, label in enumerate(labels):
        predictions[f"prob_{label}"] = probabilities[:, i]
    if len(predictions) < len(data):
        logger.warning(
            "Geneformer skipped %i of %i cells, see --forward_batch_size",
            len(data) - len(predictions),
            len(data),
        )

    rows = [{"group_by": "all", "group": "all", **metrics(predictions, labels)}]
    for column in report_by:
        for group, cells in predictions.groupby(column, sort=True):
            rows.append({"group_by": column, "group": group, **metrics(cells, labels)})
    table = pd.DataFrame(rows)
    logger.info("Metrics:\n%s", table.to_string(index=False))

    confusion = pd.DataFrame(
        confusion_matrix(
            predictions["true_label"], predictions["predicted_label"], labels=labels
        ),
        index=pd.Index(labels, name="true_label"),
        columns=labels,
    )
    logger.info("Confusion matrix:\n%s", confusion)

    table.to_csv(par["output_metrics"], index=False)
    confusion.to_csv(par["output_confusion"])
    predictions.to_csv(par["output_predictions"], index=False)


if __name__ == "__main__":
    main()
