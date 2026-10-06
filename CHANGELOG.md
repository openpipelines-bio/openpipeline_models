openpipeline_models x.x.x (unreleased)

## NEW FUNCTIONALITY

* `transform/geneformer_tokenize`: added a component to tokenize raw counts into Geneformer rank value encodings, stored in `.obsm` (PR #xxx).

* `dimred/geneformer_embeddings_extract`: added a component to extract Geneformer cell embeddings from tokenized cells into `.obsm`, on a GPU when available (PR #xxx).

* `workflows/integration/geneformer_leiden`: added a workflow to embed cells with a pretrained Geneformer model, followed by neighbour calculations, leiden clustering and UMAP (PR #xxx).
