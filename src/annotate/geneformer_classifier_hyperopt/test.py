import json
import os
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
    "executable": "./target/executable/annotate/geneformer_classifier_hyperopt/geneformer_classifier_hyperopt",
    "resources_dir": "resources_test",
    "config": "./src/annotate/geneformer_classifier_hyperopt/config.vsh.yaml",
}
## VIASH END

model_dir = f"{meta['resources_dir']}/geneformer/Geneformer-V1-10M"


def make_dataset(n_train, n_eval, seed=0):
    """Short random token sequences; label 0 cells carry tokens 2-9 first, so
    the classes are separable."""
    rng = np.random.default_rng(seed)
    n_cells = n_train + n_eval
    labels = np.tile([0, 1], n_cells // 2 + 1)[:n_cells]
    input_ids = []
    for label in labels:
        length = int(rng.integers(10, 30))
        genes = rng.choice(np.arange(100, 20_000), size=length, replace=False)
        if label == 0:
            genes[:8] = np.arange(2, 10)
        input_ids.append(genes.tolist())
    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "length": [len(ids) for ids in input_ids],
            "label": labels.tolist(),
            "split": ["train"] * n_train + ["eval"] * n_eval,
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
        }
    )


@pytest.fixture
def inputs(tmp_path):
    def wrapper(n_train=48, n_eval=16):
        dataset_path = str(tmp_path / "train_search.dataset")
        make_dataset(n_train, n_eval).save_to_disk(dataset_path)
        id_class_dict = str(tmp_path / "id_class_dict.pkl")
        with open(id_class_dict, "wb") as f:
            pickle.dump({0: "disease", 1: "healthy"}, f)
        return dataset_path, id_class_dict

    return wrapper


def run_hyperopt(run_component, tmp_path, dataset_path, id_class_dict, *extra):
    out = {"trials": str(tmp_path / "trials.csv"), "best": str(tmp_path / "best.json")}
    # relative input paths, as Nextflow passes them
    run_component(
        [
            "--input",
            os.path.relpath(dataset_path),
            "--id_class_dict",
            os.path.relpath(id_class_dict),
            "--model",
            os.path.relpath(model_dir),
            "--model_version",
            "V1",
            "--output_trials",
            out["trials"],
            "--output_best",
            out["best"],
            *extra,
        ]
    )
    return out


def test_search(run_component, tmp_path, inputs):
    out = run_hyperopt(
        run_component,
        tmp_path,
        *inputs(),
        "--freeze_layers",
        "2",
        "--n_trials",
        "2",
        "--epochs",
        "1;2",
        "--lr_min",
        "1e-5",
        "--lr_max",
        "1e-4",
    )
    trials = pd.read_csv(out["trials"])
    assert len(trials) == 2
    assert list(trials.columns) == [
        "trial_id",
        "freeze_layers",
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "eval_loss",
        "eval_accuracy",
        "eval_macro_f1",
        "objective",
    ]
    assert (trials["freeze_layers"] == 2).all()
    assert trials["learning_rate"].between(1e-5, 1e-4).all()
    assert trials["weight_decay"].between(0.0, 0.3).all()
    assert set(trials["num_train_epochs"]) <= {1, 2}
    # the objective of the transformers Ray integration is accuracy + macro F1
    assert np.allclose(
        trials["objective"], trials["eval_accuracy"] + trials["eval_macro_f1"]
    )
    assert trials["objective"].is_monotonic_decreasing

    with open(out["best"]) as f:
        best = json.load(f)
    first = trials.iloc[0]
    assert best["trial_id"] == first["trial_id"]
    assert best["objective"] == pytest.approx(first["objective"])
    assert best["freeze_layers"] == 2
    args = best["training_args"]
    assert args["learning_rate"] == pytest.approx(first["learning_rate"])
    assert args["num_train_epochs"] == first["num_train_epochs"]
    assert args["lr_scheduler_type"] == "cosine"
    assert args["warmup_ratio"] == 0.1
    assert args["warmup_steps"] == 0
    assert args["per_device_train_batch_size"] == 8
    # the precision is chosen again by the training on its own device
    assert "bf16" not in args


def test_freeze_all_layers_errors(run_component, tmp_path, inputs):
    with pytest.raises(CalledProcessError) as err:
        run_hyperopt(run_component, tmp_path, *inputs(), "--freeze_layers", "6")
    assert re.search(
        r"--freeze_layers must be between 0 and 5, the model has 6 layers",
        err.value.stdout.decode("utf-8"),
    )


def test_too_few_cells_errors(run_component, tmp_path, inputs):
    with pytest.raises(CalledProcessError) as err:
        run_hyperopt(
            run_component, tmp_path, *inputs(n_train=40), "--freeze_layers", "2"
        )
    assert re.search(
        r"40 training cells with --batch_size 8 give fewer than 1 logging step",
        err.value.stdout.decode("utf-8"),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
