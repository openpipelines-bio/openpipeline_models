import json
import os
import pickle
import re
import sys
from subprocess import CalledProcessError

import numpy as np
import pytest
from datasets import Dataset
from safetensors.numpy import load_file
from transformers import AutoConfig

## VIASH START
meta = {
    "executable": "./target/executable/annotate/geneformer_classifier_train/geneformer_classifier_train",
    "resources_dir": "resources_test",
    "config": "./src/annotate/geneformer_classifier_train/config.vsh.yaml",
}
## VIASH END

model_dir = f"{meta['resources_dir']}/geneformer/Geneformer-V1-10M"
ID_CLASS_DICT = {0: "disease", 1: "healthy"}


def make_dataset(n_cells, seed=0):
    rng = np.random.default_rng(seed)
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
            "split": ["train"] * n_cells,
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
        }
    )


def make_params(freeze_layers=2):
    return {
        "trial_id": "abc",
        "freeze_layers": freeze_layers,
        "objective": 1.0,
        "training_args": {
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "warmup_steps": 0,
            "report_to": "none",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "num_train_epochs": 1,
        },
    }


@pytest.fixture
def inputs(tmp_path):
    def wrapper(n_cells=48, params=None):
        dataset_path = str(tmp_path / "train.dataset")
        make_dataset(n_cells).save_to_disk(dataset_path)
        id_class_dict = str(tmp_path / "id_class_dict.pkl")
        with open(id_class_dict, "wb") as f:
            pickle.dump(ID_CLASS_DICT, f)
        params_path = str(tmp_path / "best.json")
        with open(params_path, "w") as f:
            json.dump(params or make_params(), f)
        return dataset_path, id_class_dict, params_path

    return wrapper


def run_train(run_component, output_model, dataset_path, id_class_dict, params):
    run_component(
        [
            "--input",
            dataset_path,
            "--id_class_dict",
            id_class_dict,
            "--params",
            params,
            "--model",
            model_dir,
            "--model_version",
            "V1",
            "--output_model",
            output_model,
        ]
    )


def test_train(run_component, tmp_path, inputs):
    output_model = str(tmp_path / "finetuned")
    run_train(run_component, output_model, *inputs())

    assert sorted(os.listdir(output_model)) == [
        "config.json",
        "id_class_dict.pkl",
        "model.safetensors",
        "training_args.bin",
        "training_params.json",
    ]
    with open(f"{output_model}/config.json") as f:
        config = json.load(f)
    assert config["architectures"] == ["BertForSequenceClassification"]
    # 2 labels is the transformers default, so config.json leaves out id2label
    assert AutoConfig.from_pretrained(output_model).num_labels == 2
    with open(f"{output_model}/id_class_dict.pkl", "rb") as f:
        assert pickle.load(f) == ID_CLASS_DICT
    with open(f"{output_model}/training_params.json") as f:
        assert json.load(f)["freeze_layers"] == 2

    # the first 2 encoder layers are frozen, the others are fine-tuned
    pretrained = load_file(f"{model_dir}/model.safetensors")
    finetuned = load_file(f"{output_model}/model.safetensors")
    for layer in range(6):
        key = f"bert.encoder.layer.{layer}.attention.self.query.weight"
        unchanged = np.array_equal(pretrained[key], finetuned[key])
        assert unchanged == (layer < 2), f"layer {layer}"
    assert "classifier.weight" in finetuned


def test_freeze_too_many_layers_errors(run_component, tmp_path, inputs):
    with pytest.raises(CalledProcessError) as err:
        run_train(
            run_component,
            str(tmp_path / "finetuned"),
            *inputs(params=make_params(freeze_layers=6)),
        )
    assert re.search(
        r"--freeze_layers must be between 0 and 5", err.value.stdout.decode("utf-8")
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
