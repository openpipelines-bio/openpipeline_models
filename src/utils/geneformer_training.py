import os

# Ray Train V2 drops the trial reporting the transformers Ray integration uses;
# both must be set before ray or torch are imported.
os.environ.setdefault("RAY_TRAIN_V2_ENABLED", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

# Settings of the fine-tuning that are not searched. warmup_steps must be 0,
# otherwise TrainingArguments ignores warmup_ratio.
PINNED_TRAINING_ARGS = {
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "warmup_steps": 0,
    "report_to": "none",
}


def precision_args(mode):
    """TrainingArguments for --mixed_precision: bf16 when asked or, with
    `auto`, when the GPU supports it; fp32 otherwise."""
    supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if mode == "bf16" and not supported:
        raise ValueError("--mixed_precision bf16 needs a GPU that supports bf16")
    if mode in ("bf16", "auto") and supported:
        return {"bf16": True}
    return {}


def check_freeze_layers(model_dir, freeze_layers):
    n_layers = AutoConfig.from_pretrained(model_dir).num_hidden_layers
    if not 0 <= freeze_layers < n_layers:
        raise ValueError(
            f"--freeze_layers must be between 0 and {n_layers - 1}, the model "
            f"has {n_layers} layers; got {freeze_layers}"
        )
    return n_layers


def check_logging_steps(n_cells, batch_size):
    """The Classifier sets logging_steps to round(n_cells / batch_size / 10),
    which TrainingArguments rejects when it is 0."""
    if round(n_cells / batch_size / 10) < 1:
        raise ValueError(
            f"{n_cells} training cells with --batch_size {batch_size} give fewer "
            "than 1 logging step per epoch; the training needs more than "
            f"{5 * batch_size} cells, or a smaller --batch_size."
        )


def devices():
    """Number of GPUs, and the ngpu the Classifier divides its CPUs by."""
    n_gpus = torch.cuda.device_count()
    return n_gpus, max(n_gpus, 1)
