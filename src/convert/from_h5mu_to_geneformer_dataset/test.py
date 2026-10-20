import pickle
import re
import sys
from subprocess import CalledProcessError

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import pytest
from datasets import load_from_disk

## VIASH START
meta = {
    "executable": "./target/executable/convert/from_h5mu_to_geneformer_dataset/from_h5mu_to_geneformer_dataset",
    "resources_dir": "src/utils",
    "config": "./src/convert/from_h5mu_to_geneformer_dataset/config.vsh.yaml",
}
## VIASH END

PAD_TOKEN = 0
N_GENES = 300
# donor -> (disease cells, healthy cells, unused cells)
DONORS = {
    "d0": (20, 30, 5),
    "d1": (12, 40, 5),
    "d2": (8, 25, 5),
    "d3": (5, 35, 5),
    "d4": (3, 30, 5),
    "d5": (2, 20, 5),
    "d6": (0, 45, 5),
    "d7": (0, 30, 5),
    "d8": (0, 25, 5),
    "d9": (0, 20, 5),
}
DISEASE_MARKERS = ["gene_0", "gene_1", "gene_2"]
HEALTHY_MARKERS = ["gene_3", "gene_4", "gene_5"]


def make_h5mu(donors):
    rng = np.random.default_rng(0)
    labels, donor_ids = [], []
    for donor, n_cells in donors.items():
        for label, n in zip(["disease", "healthy", "unused"], n_cells):
            labels += [label] * n
            donor_ids += [donor] * n
    n_obs = len(labels)
    labels = np.array(labels)

    tokens = np.full((n_obs, 2048), PAD_TOKEN, dtype=np.uint16)
    lengths = rng.integers(10, 40, size=n_obs)
    for row, length in enumerate(lengths):
        tokens[row, :length] = rng.choice(
            np.arange(2, 20_000), size=length, replace=False
        )

    # log-normalized-like expression, with the markers high in their label
    expression = rng.gamma(1.0, 1.0, size=(n_obs, N_GENES))
    expression[np.ix_(labels == "disease", [0, 1, 2])] += 5
    expression[np.ix_(labels == "healthy", [3, 4, 5])] += 5

    rna = ad.AnnData(
        X=np.zeros((n_obs, N_GENES)),
        obs=pd.DataFrame(
            {
                "group": pd.Categorical(labels),
                "donor_id": donor_ids,
                "source": np.where(np.arange(n_obs) % 2 == 0, "a", "b"),
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(
            {"gene_symbol": [f"gene_{i}" for i in range(N_GENES)]},
            index=[f"ENSG{i:011d}" for i in range(N_GENES)],
        ),
        layers={"log_normalized": expression},
    )
    rna.obsm["geneformer_tokens"] = tokens
    rna.uns["geneformer_tokenize"] = {
        "model_version": "V1",
        "model_input_size": 2048,
        "special_token": False,
        "pad_token": PAD_TOKEN,
        "cls_token": -1,
        "eos_token": -1,
    }
    prot = ad.AnnData(X=np.ones((n_obs, 2)), obs=pd.DataFrame(index=rna.obs_names))
    return mu.MuData({"rna": rna, "prot": prot})


@pytest.fixture
def write_input(random_h5mu_path):
    def wrapper(donors=DONORS):
        path = random_h5mu_path()
        make_h5mu(donors).write_h5mu(path)
        return path

    return wrapper


@pytest.fixture
def run_convert(run_component, tmp_path):
    def wrapper(input_path, *extra):
        out = {
            "train": str(tmp_path / "train.dataset"),
            "train_search": str(tmp_path / "train_search.dataset"),
            "test": str(tmp_path / "test.dataset"),
            "id_class_dict": str(tmp_path / "id_class_dict.pkl"),
            "label_qc": str(tmp_path / "label_qc.csv"),
        }
        run_component(
            [
                "--input",
                input_path,
                "--model_version",
                "V1",
                "--obs_label",
                "group",
                "--output_train",
                out["train"],
                "--output_train_search",
                out["train_search"],
                "--output_test",
                out["test"],
                "--output_id_class_dict",
                out["id_class_dict"],
                "--output_label_qc",
                out["label_qc"],
                *extra,
            ]
        )
        return out

    return wrapper


def donors_of(dataset, column="donor_id"):
    return set(dataset[column])


def test_donor_split(write_input, run_convert):
    out = run_convert(
        write_input(),
        "--obs_donor",
        "donor_id",
        "--obs_report_by",
        "donor_id;source",
        "--test_size",
        "0.2",
        "--eval_size",
        "0.2",
    )
    train = load_from_disk(out["train"])
    search = load_from_disk(out["train_search"])
    test = load_from_disk(out["test"])

    with open(out["id_class_dict"], "rb") as f:
        assert pickle.load(f) == {0: "disease", 1: "healthy"}

    # the unused cells are dropped, the labelled ones are all kept
    n_labelled = sum(d + h for d, h, _ in DONORS.values())
    assert len(train) + len(test) == n_labelled
    assert set(train["split"]) == {"train", "eval"}
    assert set(test["split"]) == {"test"}

    # whole donors only: no donor in two sets
    train_df = train.to_pandas()
    sets = {
        "train": set(train_df.loc[train_df["split"] == "train", "donor_id"]),
        "eval": set(train_df.loc[train_df["split"] == "eval", "donor_id"]),
        "test": donors_of(test),
    }
    assert not sets["train"] & sets["eval"]
    assert not sets["train"] & sets["test"]
    assert not sets["eval"] & sets["test"]
    assert set.union(*sets.values()) == set(DONORS)

    # every set holds both labels
    for split in ["train", "eval"]:
        assert set(train_df.loc[train_df["split"] == split, "label"]) == {0, 1}
    assert set(test["label"]) == {0, 1}

    # without downsampling the search set is the train set
    assert len(search) == len(train)
    assert set(search.column_names) == {
        "input_ids",
        "length",
        "label",
        "split",
        "cell_id",
        "donor_id",
        "source",
    }
    assert all(len(ids) == n for ids, n in zip(test["input_ids"], test["length"]))
    assert all(PAD_TOKEN not in ids for ids in test["input_ids"])

    qc = pd.read_csv(out["label_qc"])
    assert set(qc["set"]) == {"train", "eval", "test", "train_search", "eval_search"}
    test_qc = qc[qc["set"] == "test"].set_index("label")
    assert test_qc["n_cells"].sum() == len(test)
    assert test_qc["n_donors"]["disease"] >= 1


def test_downsampling_keeps_test_set(write_input, run_convert):
    out = run_convert(
        write_input(),
        "--obs_donor",
        "donor_id",
        "--max_cells_per_label",
        "20",
    )
    search = load_from_disk(out["train_search"]).to_pandas()
    train = load_from_disk(out["train"])
    test = load_from_disk(out["test"])

    assert search.groupby("label").size().to_dict() == {0: 20, 1: 20}
    # both sets of the search keep both labels
    assert search.groupby("split")["label"].nunique().to_dict() == {
        "eval": 2,
        "train": 2,
    }
    assert set(search["cell_id"]) <= set(train["cell_id"])
    # the test set is not downsampled
    n_labelled = sum(d + h for d, h, _ in DONORS.values())
    assert len(train) + len(test) == n_labelled
    assert len(test) > 0


def test_three_disease_donors(write_input, run_convert):
    # dealing the disease donors by size puts d0 and d1 in train and d2 in
    # eval; the test set needs one too
    donors = {
        "d0": (50, 30, 0),
        "d1": (30, 30, 0),
        "d2": (20, 30, 0),
        "d3": (0, 30, 0),
    }
    out = run_convert(
        write_input(donors),
        "--obs_donor",
        "donor_id",
        "--obs_report_by",
        "donor_id",
    )
    train = load_from_disk(out["train"]).to_pandas()
    test = load_from_disk(out["test"]).to_pandas()
    disease_donors = {
        "train": set(
            train.loc[(train["split"] == "train") & (train["label"] == 0), "donor_id"]
        ),
        "eval": set(
            train.loc[(train["split"] == "eval") & (train["label"] == 0), "donor_id"]
        ),
        "test": set(test.loc[test["label"] == 0, "donor_id"]),
    }
    assert all(len(d) == 1 for d in disease_donors.values()), disease_donors
    assert disease_donors["train"] == {"d0"}


def test_too_few_disease_donors_errors(write_input, run_convert):
    donors = {"d0": (10, 30, 0), "d1": (5, 30, 0), "d2": (0, 30, 0)}
    with pytest.raises(CalledProcessError) as err:
        run_convert(write_input(donors), "--obs_donor", "donor_id")
    stdout = err.value.stdout.decode("utf-8")
    assert re.search(r"Only 2 donors hold 'disease' cells", stdout)
    assert re.search(r"--split_level cell", stdout)


def test_cell_split(write_input, run_convert):
    donors = {"d0": (10, 30, 0), "d1": (5, 30, 0), "d2": (0, 30, 0)}
    out = run_convert(write_input(donors), "--split_level", "cell")
    train = load_from_disk(out["train"]).to_pandas()
    test = load_from_disk(out["test"]).to_pandas()
    # 15 disease cells: round(0.1 x 15) = 2 test, 2 eval
    assert (test["label"] == 0).sum() == 2
    assert ((train["split"] == "eval") & (train["label"] == 0)).sum() == 2
    assert len(train) + len(test) == 105


def test_donor_split_needs_obs_donor(write_input, run_convert):
    with pytest.raises(CalledProcessError) as err:
        run_convert(write_input())
    assert re.search(
        r"--split_level donor needs --obs_donor", err.value.stdout.decode("utf-8")
    )


def test_marker_qc(write_input, run_convert):
    out = run_convert(
        write_input(),
        "--obs_donor",
        "donor_id",
        "--disease_marker_genes",
        ";".join(DISEASE_MARKERS + ["NOT_A_GENE"]),
        "--healthy_marker_genes",
        ";".join(HEALTHY_MARKERS),
        "--var_gene_names",
        "gene_symbol",
        "--marker_layer",
        "log_normalized",
    )
    qc = pd.read_csv(out["label_qc"])
    for split in ["train", "eval", "test"]:
        means = qc[qc["set"] == split].set_index("label")
        assert (
            means.loc["disease", "mean_disease_marker_score"]
            > means.loc["healthy", "mean_disease_marker_score"] + 2
        )
        assert (
            means.loc["healthy", "mean_healthy_marker_score"]
            > means.loc["disease", "mean_healthy_marker_score"] + 2
        )


def test_missing_label_errors(write_input, run_convert):
    with pytest.raises(CalledProcessError) as err:
        run_convert(
            write_input(),
            "--obs_donor",
            "donor_id",
            "--labels",
            "disease;control",
        )
    assert re.search(
        r"Labels \['control'\] not found", err.value.stdout.decode("utf-8")
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
