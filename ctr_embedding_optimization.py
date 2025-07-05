import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def generate_data(n_samples=10000, n_features=4, embed_dim=16, random_state=42):
    """Generate synthetic user features, item embeddings and click labels."""
    rng = np.random.RandomState(random_state)
    user_features = rng.rand(n_samples, n_features)
    item_embeds = rng.normal(size=(n_samples, embed_dim))
    # ground truth weights for synthetic CTR generation
    user_w = rng.randn(n_features)
    item_w = rng.randn(embed_dim)
    logits = np.dot(user_features, user_w) + np.dot(item_embeds, item_w)
    prob = 1 / (1 + np.exp(-logits))
    clicks = rng.binomial(1, prob)
    return user_features, item_embeds, clicks


def optimize_embeddings_mlp(embeds, embed_dim=8, random_state=42):
    """Optimize embeddings via a simple autoencoder built with MLPRegressor."""
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, embed_dim),
        activation="relu",
        max_iter=50,
        random_state=random_state,
    )
    mlp.fit(embeds, embeds)
    optimized = mlp.predict(embeds)
    return optimized, mlp


def optimize_embeddings_pca(embeds, embed_dim=8, random_state=42):
    pca = PCA(n_components=embed_dim, random_state=random_state)
    optimized = pca.fit_transform(embeds)
    return optimized, pca


def train_ctr_model(train_X, train_y, val_X, val_y, epochs=10, random_state=42):
    """Train CTR model with SGD logistic regression and track AUC over epochs."""
    clf = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        random_state=random_state,
        warm_start=True,
    )
    auc_history = []
    for _ in range(epochs):
        clf.fit(train_X, train_y)
        val_pred = clf.predict_proba(val_X)[:, 1]
        auc = roc_auc_score(val_y, val_pred)
        auc_history.append(auc)
    return clf, auc_history


def plot_auc_curve(auc_history, output_path="auc_curve.png"):
    plt.figure()
    plt.plot(range(1, len(auc_history) + 1), auc_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title("CTR Training AUC Curve")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def visualize_embeddings(original, optimized, output_path="embed_scatter.png"):
    pca = PCA(n_components=2, random_state=42)
    orig_2d = pca.fit_transform(original)
    opt_2d = pca.transform(optimized)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], s=2)
    plt.title("Original Embeddings")
    plt.subplot(1, 2, 2)
    plt.scatter(opt_2d[:, 0], opt_2d[:, 1], s=2, color="orange")
    plt.title("Optimized Embeddings")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    user_feat, item_embeds, labels = generate_data()
    # split data
    X_train_val, X_test, y_train_val, y_test, item_train_val, item_test = (
        train_test_split(
            np.hstack([user_feat, item_embeds]),
            labels,
            item_embeds,
            test_size=0.2,
            random_state=42,
        )
    )
    X_train, X_val, y_train, y_val, item_train, item_val = train_test_split(
        X_train_val, y_train_val, item_train_val, test_size=0.25, random_state=42
    )

    # Optimization using PCA
    item_train_pca, pca_model = optimize_embeddings_pca(item_train)
    item_val_pca = pca_model.transform(item_val)
    item_test_pca = pca_model.transform(item_test)

    # train CTR with PCA embeddings
    train_X_pca = np.hstack([X_train[:, : user_feat.shape[1]], item_train_pca])
    val_X_pca = np.hstack([X_val[:, : user_feat.shape[1]], item_val_pca])
    test_X_pca = np.hstack([X_test[:, : user_feat.shape[1]], item_test_pca])

    clf_pca, auc_hist_pca = train_ctr_model(train_X_pca, y_train, val_X_pca, y_val)
    test_pred_pca = clf_pca.predict_proba(test_X_pca)[:, 1]

    # Optimization using MLP autoencoder
    item_train_mlp, mlp_model = optimize_embeddings_mlp(item_train)
    item_val_mlp = mlp_model.predict(item_val)
    item_test_mlp = mlp_model.predict(item_test)

    train_X_mlp = np.hstack([X_train[:, : user_feat.shape[1]], item_train_mlp])
    val_X_mlp = np.hstack([X_val[:, : user_feat.shape[1]], item_val_mlp])
    test_X_mlp = np.hstack([X_test[:, : user_feat.shape[1]], item_test_mlp])

    clf_mlp, auc_hist_mlp = train_ctr_model(train_X_mlp, y_train, val_X_mlp, y_val)
    test_pred_mlp = clf_mlp.predict_proba(test_X_mlp)[:, 1]

    # Save results
    pd.DataFrame(item_test_mlp).to_csv("optimized_item_embeds_test.csv", index=False)
    pd.DataFrame({"pctr": test_pred_mlp}).to_csv("test_pred_pctr.csv", index=False)

    # Visualizations
    plot_auc_curve(auc_hist_mlp)
    visualize_embeddings(item_train, item_train_mlp)


if __name__ == "__main__":
    main()
