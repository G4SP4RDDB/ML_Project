import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from knn import KNN
from utils import accuracy_fn, macrof1_fn, normalize_fn


def KFold_cross_validation_KNN(X, Y, K, k, seed=100):

    N = X.shape[0]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    X, Y = X[perm], Y[perm]

    split_size = N // K
    accuracies, f1_scores = [], []

    for fold_ind in range(K):
        all_ind = np.arange(N)
        val_ind = all_ind[fold_ind * split_size: (fold_ind + 1) * split_size]
        train_ind = np.setdiff1d(all_ind, val_ind, assume_unique=True)

        X_train_fold, Y_train_fold = X[train_ind, :], Y[train_ind]
        X_val_fold, Y_val_fold = X[val_ind, :], Y[val_ind]

        means = X_train_fold.mean(axis=0, keepdims=True)
        stds = X_train_fold.std(axis=0, keepdims=True) + 1e-8
        X_train_fold = normalize_fn(X_train_fold, means, stds)
        X_val_fold = normalize_fn(X_val_fold, means, stds)

        model = KNN(k=k, task_kind="classification")
        model.fit(X_train_fold, Y_train_fold)
        Y_val_fold_pred = model.predict(X_val_fold)


        accuracies.append(accuracy_fn(Y_val_fold_pred, Y_val_fold) / 100.)
        f1_scores.append(macrof1_fn(Y_val_fold_pred, Y_val_fold))

    return (np.mean(accuracies), np.std(accuracies),
            np.mean(f1_scores), np.std(f1_scores))


def run_cv_for_hyperparam(X, Y, K, k_list, seed=100):
    mean_acc, std_acc = [], []
    mean_f1, std_f1 = [], []

    for k in k_list:
        m_acc, s_acc, m_f1, s_f1 = KFold_cross_validation_KNN(
            X, Y, K, k, seed=seed)
        mean_acc.append(m_acc)
        std_acc.append(s_acc)
        mean_f1.append(m_f1)
        std_f1.append(s_f1)

    return (np.array(mean_acc), np.array(std_acc),
            np.array(mean_f1), np.array(std_f1))


def plot_cv_results(k_values, mean_acc, std_acc, mean_f1, std_f1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for ax, scores, stds, label, color in [
        (ax1, mean_acc, std_acc, "Accuracy", "steelblue"),
        (ax2, mean_f1, std_f1, "F1-score (macro)", "darkorange"),
    ]:
        best_idx = int(np.argmax(scores))
        best_k = k_values[best_idx]

        ax.plot(k_values, scores, marker='o', color=color,
                linewidth=2, label=f"Mean {label}")
        ax.fill_between(k_values, scores - stds, scores + stds,
                        alpha=0.2, color=color, label="±1 std")
        ax.axvline(x=best_k, color='red', linestyle='--',
                   label=f"Best k={best_k}")
        ax.set_xlabel("k (neighbor #)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"kNN Cross-Validation: {label} vs k", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        print(f"Best k ({label}) = {best_k},  {label} = {scores[best_idx]:.4f}")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = np.load("../../data/features.npz")
    #print("data files:", data.files) prints: "data files: ['xtrain', 'xtest', 'ytrainreg', 'ytestreg', 'ytrainclassif', 'ytestclassif']"

    X = data["xtrain"]
    y = data["ytrainclassif"]

    k_values = list(range(1, 20))

    mean_acc, std_acc, mean_f1, std_f1 = run_cv_for_hyperparam(
        X, y, K=5, k_list=k_values, seed= np.random.randint(1000))

    plot_cv_results(k_values, mean_acc, std_acc, mean_f1, std_f1)