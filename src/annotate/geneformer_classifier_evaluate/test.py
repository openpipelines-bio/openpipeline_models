import pickle
import re
import sys
from subprocess import CalledProcessError

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

## VIASH START
meta = {
    "executable": "./target/executable/annotate/geneformer_classifier_evaluate/geneformer_classifier_evaluate",
    "resources_dir": "resources_test",
    "config": "./src/annotate/geneformer_classifier_evaluate/config.vsh.yaml",
}
## VIASH END

model_dir = f"{meta['resources_dir']}/geneformer/Geneformer-V1-10M"
ID_CLASS_DICT = {0: "disease", 1: "healthy"}


def make_dataset(n_cells, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n_cells // 2 + 1)[:n_cells]
    input_ids = [
        rng.choice(
            np.arange(2, 20_000), size=int(rng.integers(10, 30)), replace=False
        ).tolist()
        for _ in range(n_cells)
    ]
    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "length": [len(ids) for ids in input_ids],
            "label": labels.tolist(),
            "split": ["test"] * n_cells,
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            # d2 holds healthy cells only
            "donor_id": [
                ["d0", "d1", "d1", "d2"][i % 4]
                if labels[i] == 1
                else ["d0", "d1"][i % 2]
                for i in range(n_cells)
            ],
        }
    )


@pytest.fixture
def finetuned_model_dir(tmp_path):
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    path = tmp_path / "finetuned"
    model.save_pretrained(path)
    return str(path)


@pytest.fixture
def inputs(tmp_path):
    def wrapper(n_cells):
        dataset_path = str(tmp_path / "test.dataset")
        make_dataset(n_cells).save_to_disk(dataset_path)
        id_class_dict = str(tmp_path / "id_class_dict.pkl")
        with open(id_class_dict, "wb") as f:
            pickle.dump(ID_CLASS_DICT, f)
        return dataset_path, id_class_dict

    return wrapper


def run_evaluate(run_component, tmp_path, model, dataset_path, id_class_dict, *extra):
    out = {
        name: str(tmp_path / f"{name}.csv")
        for name in ["metrics", "confusion", "predictions"]
    }
    run_component(
        [
            "--input",
            dataset_path,
            "--id_class_dict",
            id_class_dict,
            "--model",
            model,
            "--model_version",
            "V1",
            "--forward_batch_size",
            "8",
            "--output_metrics",
            out["metrics"],
            "--output_confusion",
            out["confusion"],
            "--output_predictions",
            out["predictions"],
            *extra,
        ]
    )
    return {name: pd.read_csv(path) for name, path in out.items()}


def test_evaluate(run_component, tmp_path, finetuned_model_dir, inputs):
    out = run_evaluate(
        run_component,
        tmp_path,
        finetuned_model_dir,
        *inputs(24),
        "--obs_report_by",
        "donor_id",
    )
    predictions, metrics, confusion = (
        out["predictions"],
        out["metrics"],
        out["confusion"],
    )

    assert len(predictions) == 24
    assert set(predictions["cell_id"]) == {f"cell_{i}" for i in range(24)}
    assert np.allclose(predictions[["prob_disease", "prob_healthy"]].sum(axis=1), 1)
    expected_true = np.where(
        predictions["cell_id"].str.split("_").str[1].astype(int) % 2 == 0,
        "disease",
        "healthy",
    )
    assert (predictions["true_label"] == expected_true).all()

    overall = metrics[metrics["group_by"] == "all"].iloc[0]
    assert overall["n_cells"] == 24
    assert overall["accuracy"] == pytest.approx(
        (predictions["true_label"] == predictions["predicted_label"]).mean()
    )
    assert overall["n_disease"] == 12
    assert 0 <= overall["roc_auc"] <= 1

    per_donor = metrics[metrics["group_by"] == "donor_id"].set_index("group")
    assert list(per_donor.index) == ["d0", "d1", "d2"]
    assert per_donor["n_cells"].sum() == 24
    # one label only: no ROC AUC
    assert np.isnan(per_donor.loc["d2", "roc_auc"])

    assert list(confusion["true_label"]) == ["disease", "healthy"]
    assert confusion[["disease", "healthy"]].to_numpy().sum() == 24


def test_last_single_cell_is_skipped(
    run_component, tmp_path, finetuned_model_dir, inputs
):
    # 25 cells in batches of 8: Geneformer skips the last one
    out = run_evaluate(run_component, tmp_path, finetuned_model_dir, *inputs(25))
    assert len(out["predictions"]) == 24
    assert out["metrics"].loc[0, "n_cells"] == 24


def test_missing_report_column_errors(
    run_component, tmp_path, finetuned_model_dir, inputs
):
    with pytest.raises(CalledProcessError) as err:
        run_evaluate(
            run_component,
            tmp_path,
            finetuned_model_dir,
            *inputs(24),
            "--obs_report_by",
            "source",
        )
    assert re.search(
        r"Columns \['source'\] not found in --input", err.value.stdout.decode("utf-8")
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
