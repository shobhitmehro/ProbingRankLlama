import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from generate_labels import compute_metrics
from sequences import load_ms_marco_data


n_queries = 98
n_docs = 50
n_layers = 32
n_dim = 4096

USE_PCA = False
PCA_COMPONENTS = 256

NORMALIZE_LABELS = False


query_set = load_ms_marco_data(n_queries, n_docs)
feature_set = compute_metrics(query_set)

context_neurons = []

for layer in range(n_layers - 1, n_layers):

    print(f"\nProcessing Layer {layer}")

    feature_names = list(next(iter(feature_set.values()))[0].keys())

    for feature in feature_names:

        print(f"\nFeature: {feature}")

        
        activations = np.zeros((n_queries * n_docs, n_dim), dtype=float)
        labels = np.zeros(n_queries * n_docs, dtype=float)

        for i in range(n_queries):
            for j in range(n_docs):
                idx = i * n_docs + j
                path = f'91activations/q{i}/d{j}layer_{layer}_activations.pt'

                if os.path.exists(path):
                    act = torch.load(path).cpu().numpy()
                    activations[idx] = act
                else:
                    print(f"Missing: {path}")
                    continue

       
        for i, query in enumerate(query_set):
            metrics = feature_set[query]
            for j in range(n_docs):
                labels[i * n_docs + j] = metrics[j][feature]

        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=0.25, random_state=42
        )

       
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

       
        if USE_PCA:
            pca = PCA(n_components=PCA_COMPONENTS)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        
        if NORMALIZE_LABELS:
            y_mean = y_train.mean()
            y_std = y_train.std() + 1e-8
            y_train = (y_train - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std

        # =============================
        # 🔹 RIDGE REGRESSION
        # =============================
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train, y_train)

        y_pred = ridge.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Ridge R2:", r2)

        if r2 > -10:
            context_neurons.append({
                "model": "ridge",
                "layer": layer,
                "feature": feature,
                "score": r2,
                "mse": mse,
                "weights": ridge.coef_
            })

       
        elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
        elastic.fit(X_train, y_train)

        y_pred = elastic.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("ElasticNet R2:", r2)

        if r2 > -10:
            context_neurons.append({
                "model": "elastic",
                "layer": layer,
                "feature": feature,
                "score": r2,
                "mse": mse,
                "weights": elastic.coef_
            })

print(f"\nTotal context neurons found: {len(context_neurons)}")