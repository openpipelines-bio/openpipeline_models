import functools
import glob
import json
import os
import sys
import tempfile

import pandas as pd

## VIASH START
par = {
    "input": "train_search.dataset",
    "id_class_dict": "id_class_dict.pkl",
    "model": "resources_test/geneformer/Geneformer-V1-10M",
    "model_version": "V1",
    "freeze_layers": 2,
    "n_trials": 2,
    "lr_min": 5e-6,
    "lr_max": 5e-4,
    "weight_decay_min": 0.0,
    "weight_decay_max": 0.3,
    "epochs": [1, 2, 3],
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "auto",
    "seed": 42,
    "output_trials": "trials.csv",
    "output_best": "best.json",
}
meta = {"resources_dir": "src/utils", "cpus": 4, "temp_dir": None}
## VIASH END

sys.path.append(meta["resources_dir"])
from setup_logger import setup_logger  # noqa: E402

# sets the Ray and torch environment, so it must come before those imports
from geneformer_training import (  # noqa: E402
    PINNED_TRAINING_ARGS,
    check_freeze_layers,
    check_logging_steps,
    devices,
    precision_args,
)
import ray  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from geneformer import Classifier  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune.search import hyperopt as ray_hyperopt  # noqa: E402

logger = setup_logger()

SEARCHED = ["learning_rate", "weight_decay", "num_train_epochs"]
METRICS = ["eval_loss", "eval_accuracy", "eval_macro_f1", "objective"]


def patch_ray(nproc, n_gpus, seed):
    """Classifier.hyperopt_classifier calls ray.init() and HyperOptSearch()
    without arguments. Give Ray only the CPUs and GPUs of this task, so that
    one trial runs per GPU, and seed the searcher."""
    ray.init = functools.partial(
        ray.init, num_cpus=nproc, num_gpus=n_gpus, include_dashboard=False
    )
    ray_hyperopt.HyperOptSearch = functools.partial(
        ray_hyperopt.HyperOptSearch, random_state_seed=seed
    )


def read_trials(search_dir):
    """One row per trial with its settings and the metrics of its last epoch,
    from the result.json files Ray Tune writes per trial."""
    rows = []
    for path in glob.glob(f"{search_dir}/**/result.json", recursive=True):
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if not lines:
            continue
        last = lines[-1]
        row = {"trial_id": last["trial_id"], "freeze_layers": par["freeze_layers"]}
        row.update({key: last["config"][key] for key in SEARCHED})
        row.update({key: last[key] for key in METRICS})
        rows.append(row)
    trials = pd.DataFrame(
        rows, columns=["trial_id", "freeze_layers", *SEARCHED, *METRICS]
    )
    return trials.sort_values(
        ["objective", "trial_id"], ascending=[False, True]
    ).reset_index(drop=True)


def main():
    # the Ray trials run in their own working directory, so relative paths,
    # as Nextflow passes them, do not resolve there
    for key in ["input", "id_class_dict", "model"]:
        par[key] = os.path.abspath(par[key])
    if par["lr_min"] <= 0 or par["lr_min"] > par["lr_max"]:
        raise ValueError("--lr_min must be larger than 0 and at most --lr_max")
    if par["weight_decay_min"] > par["weight_decay_max"]:
        raise ValueError("--weight_decay_min must be at most --weight_decay_max")
    check_freeze_layers(par["model"], par["freeze_layers"])

    data = load_from_disk(par["input"])
    if "split" not in data.column_names:
        raise ValueError(
            "--input has no 'split' column; create it with "
            "convert/from_h5mu_to_geneformer_dataset"
        )
    splits = pd.Series(data["split"]).value_counts()
    logger.info("Cells per set: %s", splits.to_dict())
    for split in ["train", "eval"]:
        if split not in splits:
            raise ValueError(f"--input has no '{split}' cells")
    check_logging_steps(int(splits["train"]), par["batch_size"])

    nproc = int(meta["cpus"] or 1)
    n_gpus, ngpu = devices()
    logger.info("%i GPUs, %i CPUs", n_gpus, nproc)
    patch_ray(nproc, n_gpus, par["seed"])

    training_args = {
        **PINNED_TRAINING_ARGS,
        "per_device_train_batch_size": par["batch_size"],
        "per_device_eval_batch_size": par["batch_size"],
        "gradient_accumulation_steps": par["gradient_accumulation_steps"],
        "seed": par["seed"],
    }
    ray_config = {
        "learning_rate": tune.loguniform(par["lr_min"], par["lr_max"]),
        "weight_decay": tune.uniform(par["weight_decay_min"], par["weight_decay_max"]),
        "num_train_epochs": tune.choice(par["epochs"]),
    }
    classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "label", "states": "all"},
        training_args={**training_args, **precision_args(par["mixed_precision"])},
        ray_config=ray_config,
        freeze_layers=par["freeze_layers"],
        num_crossval_splits=1,
        forward_batch_size=par["batch_size"],
        model_version=par["model_version"],
        nproc=nproc,
        ngpu=ngpu,
    )

    with tempfile.TemporaryDirectory(dir=meta.get("temp_dir")) as temp_dir:
        search_dir = os.path.abspath(temp_dir)
        logger.info(
            "Running %i trials with %i frozen layers",
            par["n_trials"],
            par["freeze_layers"],
        )
        classifier.validate(
            model_directory=par["model"],
            prepared_input_data_file=par["input"],
            id_class_dict_file=par["id_class_dict"],
            output_directory=search_dir,
            output_prefix="hyperopt",
            split_id_dict={"attr_key": "split", "train": ["train"], "eval": ["eval"]},
            n_hyperopt_trials=par["n_trials"],
        )
        trials = read_trials(search_dir)

    if len(trials) != par["n_trials"]:
        raise RuntimeError(
            f"{par['n_trials']} trials were requested, {len(trials)} reported results"
        )
    logger.info("Trials, best first:\n%s", trials.to_string(index=False))
    trials.to_csv(par["output_trials"], index=False)

    best = trials.iloc[0]
    output = {
        "trial_id": best["trial_id"],
        "freeze_layers": int(par["freeze_layers"]),
        **{key: float(best[key]) for key in METRICS},
        "training_args": {
            **training_args,
            "learning_rate": float(best["learning_rate"]),
            "weight_decay": float(best["weight_decay"]),
            "num_train_epochs": int(best["num_train_epochs"]),
        },
    }
    with open(par["output_best"], "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
