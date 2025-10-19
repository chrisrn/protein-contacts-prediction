# Protein Contact Map Prediction with ESM2 Embeddings
This repository implements a pipeline to predict protein contact maps using ESM2 embeddings enriched with structural features. The pipeline includes embedding generation, dimensionality reduction, enrichment using similar proteins, and training a lightweight neural network.

## Pipeline Overview

1. Generate embeddings: Extract 320-dim ESM2 embeddings and structural features from the dataset.
2. Dimensionality reduction: Apply PCA to reduce ESM2 embeddings from 320 → 64 dimensions to save computational resources.
3. Enrichment with structural features:
   * Cosine similarity: Find top-3 neighbors using embedding similarity and concatenate their   structural features → 64-dim → 80-dim embeddings.
   * Pairwise sequence similarity (pairwise2): Find top-3 neighbors using sequence alignment   and concatenate the average of their structural features → 64-dim → 72-dim embeddings.
4. Train models: Run experiments using the different embeddings:
   * 64-dim PCA embeddings
   * 80-dim cosine-enriched embeddings
   * 72-dim pairwise-enriched embeddings (best results)
 
## Set-up environment
Create a python virtual environment and install requirements:
```bash
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/esm.git
```

## Embeddings generation
First we generate ESM2 embeddings and structural features from our dataset:
```bash
python generate_embeddings.py --data_dir protein_data --out_dir embeddings/cache --struct_out_dir embeddings/structural_features
```
2 directories are created which contain train-test folders. First one is the directory containing 320-dim ESM2 embeddings and the second one is the directory containing 4-dim structural features.

Then PCA is applied to 320-dim ESM2 embeddings:
```bash
python pca_embeddings.py --emb_dir embeddings/cache --out_dir embeddings/cache_pca --n_components 64
```
The output directory contains train-test 64-dim embeddings.

Then these PCA embeddings are enriched with structural features using cosine similarity as described in the introduction:
```bash
python enrich_emberddings_cosine.py --emb_dir embeddings/cache_pca --struct_dir embeddings/structural_features --output_dir embeddings/enriched_embeddings_cosine
```
So now the output directory containts train-test 80-dim embeddings.

We apply also another way of enrichment the 2nd one described in the introduction using pairwise sequence similarity from python Bio package:
```bash
python enrich_emberddings_pairwise.py --emb_dir embeddings/cache_pca --pdb_dir protein_data --struct_dir embeddings/structural_features --output_dir embeddings/enriched_embeddings_pairwise --max_residues 50
```
Here we use only 50 residues per protein because the computations are heavy on CPU.

After exporting all the different types of embeddings we can run the [data_exploration](https://github.com/chrisrn/protein-contacts-prediction/blob/master/src/data_exploration.ipynb) notebook which shows some statistics of our data to understand them better.

## Training - testing
The model architecture is a simple feedforward neural network using 2 FC layers (CPU-friendly model). With a hidden dim of 16 we have almost 2.5k parameters to train. So we run the training for the 3 types of embeddings to compare results. First experiment is on the 64-dim pca embeddings:
```bash
python train_from_embeddings.py --pdb_dir protein_data --emb_dir embeddings/cache_pca --max_residues 50 --hidden_dim 16 --epochs 10 --results_dir results/runs_nosim
```

Second experiment is on 80-dim enriched embeddings with structural features using cosine similarity:
```bash
python train_from_embeddings.py --pdb_dir protein_data --emb_dir embeddings/enriched_embeddings_cosine --max_residues 50 --hidden_dim 16 --epochs 10 --results_dir results/runs_cosine
```

And third experiment is on 72-dim enriched embeddings with structural features using pairwise2 sequence similarity:
```bash
python train_from_embeddings.py --pdb_dir protein_data --emb_dir embeddings/enriched_embeddings_pairwise --max_residues 50 --hidden_dim 16 --epochs 10 --results_dir results/runs_pairwise
```
We use a CPU-friendly number of residues to keep from each protein sequence and we can see the results after 10 epochs on the respective result directories (matplotlib plots, tensorboard, csv with metrics, model weights).
Finally we can see some predicted vs ground-truth contact maps in [visualize_predicted_maps](https://github.com/chrisrn/protein-contacts-prediction/blob/master/src/visualize_predicted_maps.ipynb) notebeook using the model trained on pairwise2 embeddings which is the best one.

