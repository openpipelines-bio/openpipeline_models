openpipeline_models x.x.x (unreleased)

## NEW FUNCTIONALITY

* `transform/geneformer_tokenize`: added a component to tokenize raw counts into Geneformer rank value encodings, stored in `.obsm` (PR #3).

* `dimred/geneformer_embeddings_extract`: added a component to extract Geneformer cell embeddings from tokenized cells into `.obsm`, on a GPU when available (PR #3).

* `workflows/integration/geneformer_leiden`: added a workflow to embed cells with a pretrained Geneformer model, followed by neighbour calculations, leiden clustering and UMAP (PR #4).

* `perturbation/label_cells`, `perturbation/sample_cells`, `perturbation/compute_centroids`, `perturbation/geneformer_virtual_cells`, `perturbation/similarity_shift`, `perturbation/rank_genes`: added components for an in silico single-gene knockout with Geneformer: label and select the disease and healthy cells, compute their centroids, build knockout virtual cells in token space, score their cosine similarity shift towards both centroids and rank the genes (PR #5).

* `workflows/perturbation/geneformer_insilico`: added a workflow to rank the single-gene knockouts that move disease cells towards the healthy state, scattered over batches of disease cells (PR #6).
