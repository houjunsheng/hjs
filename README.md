# hjs

This repository demonstrates CTR prediction with optimized item embeddings. The
`ctr_embedding_optimization.py` script generates synthetic data, optimizes
item embeddings using PCA and an MLP autoencoder, trains a CTR model on the
optimized features, and evaluates performance using AUC.

```bash
pip install -r requirements.txt  # install dependencies
python ctr_embedding_optimization.py
```

Running the script outputs the optimized embeddings (`optimized_item_embeds_test.csv`),
predicted probabilities (`test_pred_pctr.csv`), and AUC/embedding visualizations
(`auc_curve.png`, `embed_scatter.png`).
