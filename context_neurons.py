import os
import json
import numpy as np
import matplotlib.pyplot as plt
from generate_labels import compute_metrics
from sequences import load_ms_marco_data
from context_neurons_utils import (
    build_labels_matrix,
    evaluate_model,
    get_feature_names,
    get_models,
    load_cache,
    load_layer_activations,
    plot_all_features_for_model,
    plot_average_model_comparison,
    save_cache,
    split_and_scale,
)


n_queries = 98
n_docs = 50
n_layers = 32
n_dim = 4096
cache_path = 'new_plots/r2_cache.json'
excluded_features = {
    'min_of_term_frequency',
    'min_of_stream_length_normalized_tf',
    'min_of_tf_idf',
}

os.makedirs('new_plots', exist_ok=True)

r2_cache = load_cache(cache_path)

query_set = load_ms_marco_data(n_queries, n_docs)
feature_set = compute_metrics(query_set)
feature_names = get_feature_names(feature_set, excluded_features)
models = get_models()

context_neurons = []
r2_scores_by_layer = {
    layer: {model_name: {} for model_name in models}
    for layer in range(n_layers)
}

for layer in range(n_layers):
    print(f"\n{'='*60}")
    print(f"Processing Layer {layer}")
    print(f"{'='*60}")

    activations = load_layer_activations(layer, n_queries, n_docs, n_dim)
    labels_matrix = build_labels_matrix(feature_set, feature_names, n_queries, n_docs)

    print(f"  Loaded activations for layer {layer} ({activations.shape[0]} samples)")

    X_train, X_test, train_idx, test_idx = split_and_scale(activations)

    if False:
        y_train_mean = labels_matrix[train_idx].mean(axis=0)
        y_train_std = labels_matrix[train_idx].std(axis=0) + 1e-8
        labels_matrix[train_idx] = (labels_matrix[train_idx] - y_train_mean) / y_train_std
        labels_matrix[test_idx] = (labels_matrix[test_idx] - y_train_mean) / y_train_std

    layer_str = str(layer)
    layer_cache = r2_cache.get(layer_str, {})

    # initialize cached scores so the plot data is complete even if we skip some models
    for feature_index, feature in enumerate(feature_names):
        cached_values = layer_cache.get(feature, {})
        for model_name, r2_value in cached_values.items():
            r2_scores_by_layer[layer][model_name][feature] = r2_value

    for model_name, model in models.items():
        feature_cache_missing = [
            feature
            for feature in feature_names
            if model_name not in layer_cache.get(feature, {})
        ]

        if not feature_cache_missing:
            print(f"\n  Model: {model_name} (all cached)")
            continue

        print(f"\n  Model: {model_name} (training on {len(feature_cache_missing)} features)")
        for feature_index, feature in enumerate(feature_names):
            if feature not in feature_cache_missing:
                continue

            print(f"    Feature: {feature}")
            y_train = labels_matrix[train_idx, feature_index]
            y_test = labels_matrix[test_idx, feature_index]

            stats = evaluate_model(X_train, X_test, y_train, y_test, model)
            print(f"      {model_name.capitalize()} R²: {stats['r2']:.4f}")
            r2_scores_by_layer[layer][model_name][feature] = stats['r2']
            if stats['r2'] > -10:
                context_neurons.append({
                    'model': model_name,
                    'layer': layer,
                    'feature': feature,
                    'score': stats['r2'],
                    'mse': stats['mse'],
                    'weights': stats['coef'],
                })

            if feature not in layer_cache:
                layer_cache[feature] = {}
            layer_cache[feature][model_name] = stats['r2']
            r2_cache[layer_str] = layer_cache
            save_cache(r2_cache, cache_path)

print(f"\n{'='*60}")
print(f"Total context neurons found: {len(context_neurons)}")
print(f"{'='*60}")

print(f"\nGenerating plots...")

layers = list(range(n_layers))

plot_all_features_for_model(
    r2_scores_by_layer,
    feature_names,
    layers,
    'ridge',
    'new_plots/all_features_ridge.png',
)
print('    Saved: new_plots/all_features_ridge.png')

plot_all_features_for_model(
    r2_scores_by_layer,
    feature_names,
    layers,
    'elasticnet',
    'new_plots/all_features_elasticnet.png',
)
print('    Saved: new_plots/all_features_elasticnet.png')

plot_all_features_for_model(
    r2_scores_by_layer,
    feature_names,
    layers,
    'randomforest',
    'new_plots/all_features_randomforest.png',
)
print('    Saved: new_plots/all_features_randomforest.png')

plot_average_model_comparison(
    r2_scores_by_layer,
    feature_names,
    layers,
    'new_plots/average_model_comparison.png',
)
print('    Saved: new_plots/average_model_comparison.png')

print(f"\n{'='*60}")
print(f"All plots saved to 'new_plots' folder!")
print(f"{'='*60}\n")
