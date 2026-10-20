import numpy as np


def get_tokens(adata, obsm_input, uns_input, model_version):
    """Return the padded tokens written by transform/geneformer_tokenize, the
    number of tokens per cell and the tokenizer settings, after checking that
    they exist, match `model_version` and that no cell is empty."""
    if obsm_input not in adata.obsm:
        raise ValueError(
            f"'{obsm_input}' not found in .obsm. Tokenize the cells with "
            f"transform/geneformer_tokenize first. Available: {list(adata.obsm)}"
        )
    if uns_input not in adata.uns:
        raise ValueError(
            f"'{uns_input}' not found in .uns. Tokenize the cells with "
            f"transform/geneformer_tokenize first. Available: {list(adata.uns)}"
        )
    settings = adata.uns[uns_input]
    if settings["model_version"] != model_version:
        raise ValueError(
            f"The cells were tokenized for model version {settings['model_version']}, "
            f"but --model_version is {model_version}. Tokenize again with "
            "the model version of --model."
        )

    tokens = np.asarray(adata.obsm[obsm_input])
    lengths = (tokens != settings["pad_token"]).sum(axis=1)
    if np.any(lengths == 0):
        empty = adata.obs_names[lengths == 0][:5].tolist()
        raise ValueError(
            f"{int((lengths == 0).sum())} cells have no tokens, e.g. {empty}"
        )
    return tokens, lengths, settings
