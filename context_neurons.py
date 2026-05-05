import os
import json
import numpy as np
import matplotlib.pyplot as plt
from generate_labels import compute_metrics
from sequences import load_ms_marco_data, load_MIND_data
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


n_queries = 51
n_docs = 91
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

query_set = load_MIND_data(n_queries, n_docs)
feature_set = compute_metrics(query_set)
feature_names = get_feature_names(feature_set, excluded_features)
models = get_models(input_dim=n_dim)

context_neurons = []
r2_scores_by_layer = {
    layer: {model_name: {} for model_name in models}
    for layer in range(n_layers)
}

labels_matrix = build_labels_matrix(feature_set, feature_names, n_queries, n_docs)

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Processing Model: {model_name}")
    print(f"{'='*60}")

    for layer in range(n_layers):
        print(f"\n  Layer {layer}")

        activations = load_layer_activations(
            layer,
            n_queries,
            n_docs,
            n_dim,
            activation_dir='relevant/rankllama7b/activations',
        )
        print(f"    Loaded activations ({activations.shape[0]} samples)")

        X_train, X_test, train_idx, test_idx = split_and_scale(activations)

        if False:
            y_train_mean = labels_matrix[train_idx].mean(axis=0)
            y_train_std = labels_matrix[train_idx].std(axis=0) + 1e-8
            labels_matrix[train_idx] = (labels_matrix[train_idx] - y_train_mean) / y_train_std
            labels_matrix[test_idx] = (labels_matrix[test_idx] - y_train_mean) / y_train_std

        layer_str = str(layer)
        layer_cache = r2_cache.get(layer_str, {})

        for feature_index, feature in enumerate(feature_names):
            cached_values = layer_cache.get(feature, {})
            if model_name in cached_values:
                r2_scores_by_layer[layer][model_name][feature] = cached_values[model_name]

        feature_cache_missing = [
            feature
            for feature in feature_names
            if model_name not in layer_cache.get(feature, {})
        ]

        if not feature_cache_missing:
            print(f"    Model: {model_name} (all cached)")
            continue

        print(f"    Model: {model_name} (training on {len(feature_cache_missing)} features)")
        for feature_index, feature in enumerate(feature_names):
            if feature not in feature_cache_missing:
                continue

            print(f"      Feature: {feature}")
            y_train = labels_matrix[train_idx, feature_index]
            y_test = labels_matrix[test_idx, feature_index]

            stats = evaluate_model(X_train, X_test, y_train, y_test, model)
            print(f"        {model_name.capitalize()} R²: {stats['r2']:.4f}")
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

for model_name in models:
    output_path = f'new_plots/all_features_{model_name}.png'
    plot_all_features_for_model(
        r2_scores_by_layer,
        feature_names,
        layers,
        model_name,
        output_path,
    )
    print(f'    Saved: {output_path}')

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
