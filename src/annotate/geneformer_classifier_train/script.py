import glob
import json
import os
import shutil
import sys
import tempfile

## VIASH START
par = {
    "input": "train.dataset",
    "id_class_dict": "id_class_dict.pkl",
    "model": "resources_test/geneformer/Geneformer-V1-10M",
    "model_version": "V1",
    "params": "best.json",
    "mixed_precision": "auto",
    "output_model": "finetuned_model",
}
meta = {"resources_dir": "src/utils", "cpus": 4, "temp_dir": None}
## VIASH END

sys.path.append(meta["resources_dir"])
from setup_logger import setup_logger  # noqa: E402

# sets the torch environment, so it must come before that import
from geneformer_training import (  # noqa: E402
    check_freeze_layers,
    check_logging_steps,
    devices,
    precision_args,
)
from datasets import load_from_disk  # noqa: E402
from geneformer import Classifier  # noqa: E402

logger = setup_logger()

# what train_all_data saves, without the optimizer state of the checkpoints
MODEL_FILES = ["config.json", "model.safetensors", "training_args.bin"]


def main():
    with open(par["params"]) as f:
        params = json.load(f)
    for key in ["freeze_layers", "training_args"]:
        if key not in params:
            raise ValueError(f"--params has no '{key}'")
    training_args = params["training_args"]
    check_freeze_layers(par["model"], params["freeze_layers"])

    n_cells = len(load_from_disk(par["input"]))
    check_logging_steps(n_cells, training_args["per_device_train_batch_size"])

    nproc = int(meta["cpus"] or 1)
    n_gpus, ngpu = devices()
    logger.info(
        "Training on %i cells with %i frozen layers, %i GPUs: %s",
        n_cells,
        params["freeze_layers"],
        n_gpus,
        training_args,
    )
    classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "label", "states": "all"},
        training_args={
            **training_args,
            **precision_args(par["mixed_precision"]),
            "save_strategy": "no",
        },
        freeze_layers=params["freeze_layers"],
        num_crossval_splits=0,
        forward_batch_size=training_args["per_device_eval_batch_size"],
        model_version=par["model_version"],
        nproc=nproc,
        ngpu=ngpu,
    )

    with tempfile.TemporaryDirectory(dir=meta.get("temp_dir")) as temp_dir:
        classifier.train_all_data(
            model_directory=par["model"],
            prepared_input_data_file=par["input"],
            id_class_dict_file=par["id_class_dict"],
            output_directory=temp_dir,
            output_prefix="finetune",
        )
        saved = glob.glob(f"{temp_dir}/*_geneformer_cellClassifier_finetune")
        if len(saved) != 1:
            raise RuntimeError(f"Expected one trained model directory, found {saved}")

        os.makedirs(par["output_model"], exist_ok=True)
        for name in MODEL_FILES:
            shutil.copy(os.path.join(saved[0], name), par["output_model"])
    shutil.copy(
        par["id_class_dict"], os.path.join(par["output_model"], "id_class_dict.pkl")
    )
    with open(os.path.join(par["output_model"], "training_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Wrote the fine-tuned model to %s", par["output_model"])


if __name__ == "__main__":
    main()
