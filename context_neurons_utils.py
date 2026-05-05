import json
import os
import numpy as np
import torch
from torch import nn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SparseAutoencoderRegressor:
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        lr=1e-3,
        epochs=200,
        batch_size=128,
        sparsity_lambda=1e-4,
        device=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sparsity_lambda = sparsity_lambda
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def fit(self, X, y):
        self._build_model()

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                hidden = self.model[0](xb)
                hidden_act = self.model[1](hidden)
                out = self.model[2](hidden_act)
                mse = loss_fn(out, yb)
                sparsity = (hidden_act > 0).float().mean()
                loss = mse + self.sparsity_lambda * sparsity
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            hidden = self.model[0](X_tensor)
            hidden_act = self.model[1](hidden)
            preds = self.model[2](hidden_act)
        return preds.cpu().numpy().reshape(-1)


def clone_model(model):
    try:
        return clone(model)
    except Exception:
        return model


def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def get_feature_names(feature_set, excluded_features):
    return [
        feature
        for feature in list(next(iter(feature_set.values()))[0].keys())
        if feature not in excluded_features
    ]


def load_layer_activations(layer, n_queries, n_docs, n_dim, activation_dir='/home/shobhitmehro_umass_edu/ondemand/ProbingRankLlama/relevant/rankllama7b/activations'):
    n_samples = n_queries * n_docs
    activations = np.zeros((n_samples, n_dim), dtype=float)

    for i in range(n_queries):
        for j in range(n_docs):
            idx = i * n_docs + j
            path = f'{activation_dir}/q{i}/d{j}layer_{layer}_activations.pt'
            if os.path.exists(path):
                activations[idx] = torch.load(path, map_location='cpu').cpu().numpy()
            else:
                print(f'    Missing: {path}')

    return activations


def build_labels_matrix(feature_set, feature_names, n_queries, n_docs):
    n_samples = n_queries * n_docs
    labels_matrix = np.zeros((n_samples, len(feature_names)), dtype=float)

    for i, query in enumerate(feature_set):
        metrics = feature_set[query]
        for j in range(n_docs):
            sample_idx = i * n_docs + j
            for feature_index, feature in enumerate(feature_names):
                labels_matrix[sample_idx, feature_index] = metrics[j][feature]

    return labels_matrix


def split_and_scale(activations, train_size=0.75, random_state=42):
    sample_indices = np.arange(activations.shape[0])
    train_idx, test_idx = train_test_split(
        sample_indices, test_size=1 - train_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(activations[train_idx])
    X_test = scaler.transform(activations[test_idx])

    return X_train, X_test, train_idx, test_idx


def get_models(input_dim=4096):
    return {
        'ridge': Ridge(alpha=10.0),
        'elasticnet': ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=100000,
            tol=1e-4,
            selection='random',
            random_state=42,
        ),
        'randomforest': RandomForestRegressor(
            n_estimators=30,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
        'sparseprobing': SparseAutoencoderRegressor(
            input_dim=input_dim,
            hidden_dim=256,
            lr=1e-3,
            epochs=200,
            batch_size=128,
            sparsity_lambda=1e-4,
        ),
    }


def evaluate_model(X_train, X_test, y_train, y_test, model):
    model = clone_model(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    
    if np.any(~np.isfinite(y_pred)) or np.any(~np.isfinite(y_test)):
        print(f"        Warning: NaN/Inf detected in predictions or labels. Clipping values.")
        y_pred = np.clip(y_pred, -1e6, 1e6)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        r2 = r2_score(y_test, y_pred)
        if not np.isfinite(r2):
            r2 = -1.0
        elif r2 < -1e6:
            r2 = -1.0
    except Exception as e:
        print(f"        Error calculating R²: {e}")
        r2 = -1.0
    
    return {
        'r2': r2,
        'mse': mean_squared_error(y_test, y_pred),
        'coef': getattr(model, 'coef_', None),
    }


def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = {}
    for model_name, model in models.items():
        print(f"      Training {model_name}...")
        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train)
        print(f"      Predicting {model_name}...")
        y_pred = model_clone.predict(X_test)
        results[model_name] = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'coef': getattr(model_clone, 'coef_', None),
        }
    return results


def plot_all_features_for_model(r2_scores_by_layer, all_features, layers, model_name, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_features)))

    for feature_idx, feature in enumerate(all_features):
        scores = [r2_scores_by_layer[layer][model_name].get(feature, 0) for layer in layers]
        ax.plot(
            layers,
            scores,
            marker='o',
            label=feature,
            linewidth=1.8,
            markersize=4,
            color=colors[feature_idx],
        )

    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{model_name.capitalize()} Regression: R² Scores for All Features Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_average_model_comparison(r2_scores_by_layer, all_features, layers, output_path):
    import matplotlib.pyplot as plt

    average_scores = {}
    for model_name in next(iter(r2_scores_by_layer.values())).keys():
        average_scores[model_name] = []

    for layer in layers:
        for model_name in average_scores:
            scores = [r2_scores_by_layer[layer][model_name].get(feature, 0) for feature in all_features]
            average_scores[model_name].append(np.mean(scores))

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, scores in average_scores.items():
        ax.plot(
            layers,
            scores,
            marker='o',
            linewidth=2,
            markersize=6,
            label=model_name.capitalize(),
        )

    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('Average R² Score', fontsize=12)
    ax.set_title('Average R² Scores Across All Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
