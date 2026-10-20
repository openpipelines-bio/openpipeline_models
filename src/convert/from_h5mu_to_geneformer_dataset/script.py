import pickle
import sys

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset

## VIASH START
par = {
    "input": "labelled.h5mu",
    "modality": "rna",
    "obsm_input": "geneformer_tokens",
    "uns_input": "geneformer_tokenize",
    "model_version": "V1",
    "obs_label": "geneformer_finetune_group",
    "labels": ["disease", "healthy"],
    "split_level": "donor",
    "obs_donor": "donor_id",
    "test_size": 0.1,
    "eval_size": 0.1,
    "max_cells_per_label": 3000,
    "seed": 42,
    "obs_report_by": None,
    "disease_marker_genes": None,
    "healthy_marker_genes": None,
    "var_gene_names": None,
    "marker_layer": None,
    "output_train": "train.dataset",
    "output_train_search": "train_search.dataset",
    "output_test": "test.dataset",
    "output_id_class_dict": "id_class_dict.pkl",
    "output_label_qc": "label_qc.csv",
}
meta = {"resources_dir": "src/utils"}
## VIASH END

sys.path.append(meta["resources_dir"])
from setup_logger import setup_logger  # noqa: E402
from geneformer_tokens import get_tokens  # noqa: E402

logger = setup_logger()

SPLITS = ["train", "eval", "test"]
# dataset columns written by this component; Geneformer reserves "label"
RESERVED_COLUMNS = {"input_ids", "length", "label", "split", "cell_id"}


def check_column(obs, column, argument):
    if column not in obs.columns:
        raise ValueError(
            f"{argument} '{column}' not found in .obs. Available: {list(obs.columns)}"
        )


def split_fractions():
    test_size, eval_size = par["test_size"], par["eval_size"]
    if test_size <= 0 or eval_size <= 0 or test_size + eval_size >= 1:
        raise ValueError(
            "--test_size and --eval_size must be larger than 0 and sum to less "
            f"than 1, got {test_size} and {eval_size}"
        )
    return {"train": 1 - test_size - eval_size, "eval": eval_size, "test": test_size}


def deal(counts, fractions, assigned, split_of):
    """Assign the donors in `counts` (donor -> number of cells), in order, to
    the set that is furthest below its target number of cells. `assigned` holds
    the cells already in each set."""
    total = sum(assigned.values()) + int(counts.sum())
    targets = {split: fraction * total for split, fraction in fractions.items()}
    for donor, n_cells in counts.items():
        split = max(SPLITS, key=lambda s: (targets[s] - assigned[s], s == "train"))
        split_of[donor] = split
        assigned[split] += int(n_cells)


def donor_order(counts, rng):
    """Donors by decreasing number of cells, ties in a seeded random order."""
    shuffled = counts.iloc[rng.permutation(len(counts))]
    return shuffled.sort_values(ascending=False, kind="stable")


def split_by_donor(obs, rng):
    fractions = split_fractions()
    donor_label = pd.crosstab(obs["donor"], obs["label"])
    rare_label = donor_label.sum().idxmin()
    rare_donors = donor_label[rare_label][donor_label[rare_label] > 0]
    if len(rare_donors) < len(SPLITS):
        raise ValueError(
            f"Only {len(rare_donors)} donors hold '{rare_label}' cells; a donor "
            f"split needs at least {len(SPLITS)}, one per set. Use "
            f"--split_level cell to split the cells instead.\n{donor_label}"
        )

    # 1. donors with the rarest label, balanced on the cells of that label
    split_of = {}
    deal(
        donor_order(rare_donors, rng),
        fractions,
        {split: 0 for split in SPLITS},
        split_of,
    )
    # every set needs a donor with the rarest label: move the smallest donors
    # of the train set to the sets that ended up without one
    for split in ["test", "eval"]:
        if split not in split_of.values():
            train_donors = [d for d, s in split_of.items() if s == "train"]
            smallest = min(train_donors, key=lambda d: (rare_donors[d], d))
            split_of[smallest] = split

    # 2. the other donors, balanced on the cells of the other labels
    other_labels = donor_label.drop(columns=rare_label).sum(axis=1)
    assigned = {
        split: int(other_labels[[d for d, s in split_of.items() if s == split]].sum())
        for split in SPLITS
    }
    other_donors = other_labels[~other_labels.index.isin(list(split_of))]
    deal(donor_order(other_donors, rng), fractions, assigned, split_of)

    return obs["donor"].map(split_of).astype(str).to_numpy()


def split_by_cell(obs, rng):
    fractions = split_fractions()
    split = np.full(len(obs), "train", dtype=object)
    for label in par["labels"]:
        index = np.flatnonzero(obs["label"].to_numpy() == label)
        if len(index) < len(SPLITS):
            raise ValueError(
                f"Label '{label}' has {len(index)} cells, need at least "
                f"{len(SPLITS)}, one per set."
            )
        index = rng.permutation(index)
        n_test = max(1, round(fractions["test"] * len(index)))
        n_eval = max(1, round(fractions["eval"] * len(index)))
        split[index[:n_test]] = "test"
        split[index[n_test : n_test + n_eval]] = "eval"
    return split.astype(str)


def downsample(obs, rng):
    """Indices of at most --max_cells_per_label cells of each label over the
    train and eval sets, taken proportionally from both sets."""
    keep = []
    for label in par["labels"]:
        is_label = (obs["label"] == label).to_numpy()
        index = {
            split: np.flatnonzero(is_label & (obs["split"] == split).to_numpy())
            for split in ["train", "eval"]
        }
        n_label = len(index["train"]) + len(index["eval"])
        n_max = par["max_cells_per_label"]
        if n_label > n_max:
            n_eval = min(
                len(index["eval"]), max(1, round(n_max * len(index["eval"]) / n_label))
            )
            n_keep = {"train": n_max - n_eval, "eval": n_eval}
        else:
            n_keep = {split: len(rows) for split, rows in index.items()}
        for split, rows in index.items():
            keep.append(rng.choice(rows, size=n_keep[split], replace=False))
    return np.sort(np.concatenate(keep))


def marker_scores(adata):
    """Mean expression of each marker gene list minus a reference set, per
    cell, with scanpy.tl.score_genes. Returns {score name: scores}."""
    gene_lists = {
        "disease_marker_score": par["disease_marker_genes"],
        "healthy_marker_score": par["healthy_marker_genes"],
    }
    gene_lists = {name: genes for name, genes in gene_lists.items() if genes}
    if not gene_lists:
        return {}

    if par["marker_layer"]:
        if par["marker_layer"] not in adata.layers:
            raise ValueError(
                f"--marker_layer '{par['marker_layer']}' not found in .layers. "
                f"Available: {list(adata.layers)}"
            )
        X = adata.layers[par["marker_layer"]]
    else:
        X = adata.X
    if par["var_gene_names"]:
        if par["var_gene_names"] not in adata.var.columns:
            raise ValueError(
                f"--var_gene_names '{par['var_gene_names']}' not found in .var. "
                f"Available: {list(adata.var.columns)}"
            )
        gene_names = adata.var[par["var_gene_names"]].astype(str).to_numpy()
    else:
        gene_names = adata.var_names.to_numpy()

    expression = ad.AnnData(X=X, obs=pd.DataFrame(index=adata.obs_names))
    expression.var_names = gene_names
    expression.var_names_make_unique()

    scores = {}
    for name, genes in gene_lists.items():
        found = [gene for gene in genes if gene in expression.var_names]
        missing = sorted(set(genes) - set(found))
        if not found:
            raise ValueError(f"None of the genes for {name} are in the data: {genes}")
        if missing:
            logger.warning("Genes for %s not in the data: %s", name, missing)
        sc.tl.score_genes(
            expression,
            found,
            score_name=name,
            random_state=par["seed"],
            use_raw=False,
        )
        scores[name] = expression.obs[name].to_numpy()
        logger.info("Scored %i genes for %s", len(found), name)
    return scores


def to_dataset(tokens, lengths, obs, report, index):
    columns = {
        "input_ids": [tokens[i, : lengths[i]].tolist() for i in index],
        "length": lengths[index].tolist(),
        "label": obs["label_id"].to_numpy()[index].tolist(),
        "split": obs["split"].to_numpy()[index].tolist(),
        "cell_id": obs.index.to_numpy()[index].tolist(),
    }
    for column in par["obs_report_by"] or []:
        columns[column] = report[column].astype(str).to_numpy()[index].tolist()
    return Dataset.from_dict(columns)


def label_qc(obs, search_index, score_names):
    sets = {split: obs[obs["split"] == split] for split in SPLITS}
    search = obs.iloc[search_index]
    sets["train_search"] = search[search["split"] == "train"]
    sets["eval_search"] = search[search["split"] == "eval"]

    rows = []
    for name, cells in sets.items():
        for label in par["labels"]:
            label_cells = cells[cells["label"] == label]
            row = {"set": name, "label": label, "n_cells": len(label_cells)}
            if "donor" in obs.columns:
                row["n_donors"] = label_cells["donor"].nunique()
            for score in score_names:
                row[f"mean_{score}"] = label_cells[score].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    if len(par["labels"]) < 2 or len(set(par["labels"])) != len(par["labels"]):
        raise ValueError(
            f"--labels must hold at least 2 distinct values, got {par['labels']}"
        )
    report_by = par["obs_report_by"] or []
    reserved = RESERVED_COLUMNS.intersection(report_by)
    if reserved:
        raise ValueError(
            f"--obs_report_by cannot hold the reserved names {sorted(reserved)}"
        )
    if par["split_level"] == "donor" and not par["obs_donor"]:
        raise ValueError("--split_level donor needs --obs_donor")

    logger.info("Reading modality '%s' from %s", par["modality"], par["input"])
    adata = mu.read_h5ad(par["input"], mod=par["modality"])
    check_column(adata.obs, par["obs_label"], "--obs_label")
    for column in report_by:
        check_column(adata.obs, column, "--obs_report_by")
    if par["obs_donor"]:
        check_column(adata.obs, par["obs_donor"], "--obs_donor")

    labelled = adata.obs[par["obs_label"]].astype(str).isin(par["labels"]).to_numpy()
    adata = adata[labelled].copy()
    tokens, lengths, _ = get_tokens(
        adata, par["obsm_input"], par["uns_input"], par["model_version"]
    )

    obs = pd.DataFrame(index=adata.obs_names)
    obs["label"] = adata.obs[par["obs_label"]].astype(str).to_numpy()
    counts = obs["label"].value_counts()
    absent = [label for label in par["labels"] if label not in counts.index]
    if absent:
        raise ValueError(
            f"Labels {absent} not found in .obs['{par['obs_label']}']. "
            f"Found: {counts.to_dict()}"
        )
    logger.info("%i labelled cells: %s", len(obs), counts.to_dict())
    class_id = {label: i for i, label in enumerate(par["labels"])}
    obs["label_id"] = obs["label"].map(class_id).astype(int)
    if par["obs_donor"]:
        obs["donor"] = adata.obs[par["obs_donor"]].astype(str).to_numpy()

    rng = np.random.default_rng(par["seed"])
    if par["split_level"] == "donor":
        obs["split"] = split_by_donor(obs, rng)
    else:
        obs["split"] = split_by_cell(obs, rng)

    split_label = pd.crosstab(obs["split"], obs["label"]).reindex(
        index=SPLITS, columns=par["labels"], fill_value=0
    )
    logger.info("Cells per set and label:\n%s", split_label)
    if "donor" in obs.columns:
        donors = (
            obs.groupby(["split", "label"])["donor"].nunique().unstack(fill_value=0)
        )
        logger.info("Donors per set and label:\n%s", donors)
        shared = obs.groupby("donor")["split"].nunique()
        if par["split_level"] == "donor" and (shared > 1).any():
            raise RuntimeError(
                f"Donors in more than one set: {list(shared[shared > 1].index)}"
            )
    if (split_label == 0).to_numpy().any():
        raise ValueError(f"Every set needs cells of every label:\n{split_label}")

    scores = marker_scores(adata)
    for name, values in scores.items():
        obs[name] = values

    search_index = downsample(obs, rng)
    index = {split: np.flatnonzero(obs["split"] == split) for split in SPLITS}
    train_index = np.sort(np.concatenate([index["train"], index["eval"]]))

    for path, rows in [
        (par["output_train"], train_index),
        (par["output_train_search"], search_index),
        (par["output_test"], index["test"]),
    ]:
        logger.info("Writing %i cells to %s", len(rows), path)
        to_dataset(tokens, lengths, obs, adata.obs, rows).save_to_disk(path)

    with open(par["output_id_class_dict"], "wb") as f:
        pickle.dump({i: label for label, i in class_id.items()}, f)

    qc = label_qc(obs, search_index, list(scores))
    logger.info("Label QC:\n%s", qc.to_string(index=False))
    qc.to_csv(par["output_label_qc"], index=False)


if __name__ == "__main__":
    main()
