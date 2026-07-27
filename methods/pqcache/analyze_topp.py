#!/usr/bin/env python3
"""
Analyze the structure of qwen_topp_09_0320_1.txt:
- grouped by dataset
- each entry has length and layer info (28 layers x N heads)
- compute various statistics and draw heatmaps
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_topp_file(filepath):
    """Parse TOPP file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    datasets = {}
    current_dataset = None
    current_data = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # check if it's a dataset name (not starting with data:X, length, layer-)
        if not line.startswith('data:') and not line.startswith('length:') and not line.startswith('layer-'):
            current_dataset = line
            if current_dataset not in datasets:
                datasets[current_dataset] = []
            continue

        # check if it's the start of a data entry
        if line.startswith('data:'):
            data_idx = int(line.split(':')[1])
            current_data = {'idx': data_idx, 'layers': {}}
            datasets[current_dataset].append(current_data)
            continue

        # check if it's length
        if line.startswith('length:'):
            length = int(line.split(':')[1])
            current_data['length'] = length
            continue

        # check if it's a layer
        if line.startswith('layer-'):
            match = re.match(r'layer-(\d+):(.+)', line)
            if match and current_data:
                layer_idx = int(match.group(1))
                values_str = match.group(2)

                # parse nested list, extract all numbers
                numbers = re.findall(r'\d+', values_str)
                values = [int(n) for n in numbers]

                # should have N head values
                current_data['layers'][layer_idx] = values

    return datasets


def analyze_dataset(datasets, dataset_name):
    """Analyze a single dataset"""
    data_list = datasets[dataset_name]

    if not data_list:
        return None

    # dynamically get number of layers and heads
    # from the layers of the first entry
    all_layers = set()
    all_heads = set()
    for data in data_list:
        for layer_idx, values in data['layers'].items():
            all_layers.add(layer_idx)
            all_heads.update(range(len(values)))

    num_layers = max(all_layers) + 1 if all_layers else 28
    num_heads = max(all_heads) + 1 if all_heads else 28

    # store stats for each entry
    data_stats = []

    for data in data_list:
        length = data.get('length', 0)
        layers = data['layers']

        # per-entry layer average token
        layer_avg_tokens = []
        # per-entry head average token
        head_avg_tokens = []

        for layer_idx in range(num_layers):
            if layer_idx in layers:
                layer_values = layers[layer_idx]
                layer_avg = np.mean(layer_values)
                layer_avg_tokens.append(layer_avg)
            else:
                layer_avg_tokens.append(0)

        # compute average per head
        for head_idx in range(num_heads):
            head_values = []
            for layer_idx in range(num_layers):
                if layer_idx in layers:
                    head_values.append(layers[layer_idx][head_idx])
            head_avg_tokens.append(np.mean(head_values) if head_values else 0)

        # compute average token for this entry
        all_tokens = []
        for layer_idx in range(num_layers):
            if layer_idx in layers:
                all_tokens.extend(layers[layer_idx])

        data_avg_token = np.mean(all_tokens) if all_tokens else 0
        data_avg_ratio = data_avg_token / length if length > 0 else 0

        data_stats.append({
            'length': length,
            'avg_token': data_avg_token,
            'avg_ratio': data_avg_ratio,
            'layer_avg_tokens': layer_avg_tokens,
            'head_avg_tokens': head_avg_tokens,
            'layers': layers
        })

    # compute overall stats
    total_length = np.mean([d['length'] for d in data_stats])
    total_avg_token = np.mean([d['avg_token'] for d in data_stats])
    total_avg_ratio = total_avg_token / total_length if total_length > 0 else 0

    # average token per layer
    layer_mean_tokens = []
    for layer_idx in range(num_layers):
        layer_tokens = [d['layer_avg_tokens'][layer_idx] for d in data_stats]
        layer_mean_tokens.append(np.mean(layer_tokens))

    # average token per head per layer (28xN)
    layer_head_mean_tokens = []
    for layer_idx in range(num_layers):
        layer_vals = []
        for head_idx in range(num_heads):
            head_tokens = []
            for d in data_stats:
                if layer_idx in d['layers']:
                    head_tokens.append(d['layers'][layer_idx][head_idx])
            layer_vals.append(np.mean(head_tokens) if head_tokens else 0)
        layer_head_mean_tokens.append(layer_vals)

    # average token/length per layer
    layer_mean_ratios = []
    for layer_idx in range(num_layers):
        layer_ratios = [d['layers'].get(layer_idx, [0]*num_heads) for d in data_stats]
        layer_ratio = []
        for layer_vals in layer_ratios:
            if layer_vals:
                layer_ratio.extend(layer_vals)
        if layer_ratio and total_length > 0:
            layer_mean_ratios.append(np.mean(layer_ratio) / total_length)
        else:
            layer_mean_ratios.append(0)

    # average token/length per head per layer (28xN)
    layer_head_mean_ratios = []
    for layer_idx in range(num_layers):
        layer_vals = []
        for head_idx in range(num_heads):
            head_vals = []
            for d in data_stats:
                if layer_idx in d['layers']:
                    val = d['layers'][layer_idx][head_idx]
                    if d['length'] > 0:
                        head_vals.append(val / d['length'])
            layer_vals.append(np.mean(head_vals) if head_vals else 0)
        layer_head_mean_ratios.append(layer_vals)

    # average token/length per entry
    data_mean_ratios = [d['avg_ratio'] for d in data_stats]

    # token/length per layer (averaged over entries)
    layer_data_ratios = []
    for layer_idx in range(num_layers):
        layer_ratios = []
        for d in data_stats:
            if layer_idx in d['layers']:
                layer_vals = d['layers'][layer_idx]
                length = d['length']
                if length > 0:
                    layer_ratios.append(np.mean(layer_vals) / length)
        layer_data_ratios.append(np.mean(layer_ratios) if layer_ratios else 0)

    # token/length per head per layer (averaged over entries) (28xN)
    # same as layer_head_mean_ratios, only computed differently, here averaged over entries
    layer_head_data_ratios = layer_head_mean_ratios  # reuse the result above

    return {
        'total_length': total_length,
        'total_avg_token': total_avg_token,
        'total_avg_ratio': total_avg_ratio,
        'num_data': len(data_stats),
        'layer_mean_tokens': layer_mean_tokens,
        'layer_head_mean_tokens': layer_head_mean_tokens,  # 28xN
        'layer_mean_ratios': layer_mean_ratios,
        'layer_head_mean_ratios': layer_head_mean_ratios,  # 28xN
        'data_mean_ratios': data_mean_ratios,
        'layer_data_ratios': layer_data_ratios,
        'layer_head_data_ratios': layer_head_data_ratios,  # 28xN
        'data_stats': data_stats
    }


def create_heatmap(data, title, save_path, xlabel='Head', ylabel='Layer', cmap='YlOrRd'):
    """Create heatmap"""
    fig, ax = plt.subplots(figsize=(10, 12))

    # convert to numpy array
    data_array = np.array(data)

    sns.heatmap(data_array, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=[f'H{i}' for i in range(data_array.shape[1])],
                yticklabels=[f'L{i}' for i in range(data_array.shape[0])],
                ax=ax, cbar_kws={'label': 'Token Count'})

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_layer_head_heatmap(dataset_stats, dataset_name, output_dir):
    """Create layer x head heatmap (average token count per head per layer)"""
    # dynamically get number of layers and heads
    layer_head_tokens = dataset_stats['layer_head_mean_tokens']
    num_layers = len(layer_head_tokens) if layer_head_tokens else 28
    num_heads = len(layer_head_tokens[0]) if layer_head_tokens else 28

    # build N_layers x N_heads matrix
    matrix = []
    for layer_idx in range(num_layers):
        layer_vals = []
        for head_idx in range(num_heads):
            # compute the average of this layer and head over all entries
            head_vals = []
            for d in dataset_stats['data_stats']:
                if layer_idx in d['layers']:
                    head_vals.append(d['layers'][layer_idx][head_idx])
            layer_vals.append(np.mean(head_vals) if head_vals else 0)
        matrix.append(layer_vals)

    title = f'{dataset_name}: Average Token Count per Layer and Head'
    save_path = os.path.join(output_dir, f'{dataset_name}_layer_head_heatmap.png')
    create_heatmap(matrix, title, save_path)

    return matrix


def create_ratio_heatmap(dataset_stats, dataset_name, output_dir):
    """Create layer x head heatmap (token/length ratio)"""
    # dynamically get number of layers and heads
    layer_head_ratios = dataset_stats['layer_head_mean_ratios']
    num_layers = len(layer_head_ratios) if layer_head_ratios else 28
    num_heads = len(layer_head_ratios[0]) if layer_head_ratios else 28

    matrix = []
    for layer_idx in range(num_layers):
        layer_vals = []
        for head_idx in range(num_heads):
            head_vals = []
            for d in dataset_stats['data_stats']:
                if layer_idx in d['layers']:
                    val = d['layers'][layer_idx][head_idx]
                    if d['length'] > 0:
                        head_vals.append(val / d['length'])
            layer_vals.append(np.mean(head_vals) if head_vals else 0)
        matrix.append(layer_vals)

    title = f'{dataset_name}: Token/Length Ratio per Layer and Head'
    save_path = os.path.join(output_dir, f'{dataset_name}_layer_head_ratio_heatmap.png')
    create_heatmap(matrix, title, save_path, cmap='Blues')

    return matrix


def create_summary_plots(dataset_stats, dataset_name, output_dir):
    """Create summary statistics plots"""
    # get number of layers and heads from the data
    layer_head_tokens = dataset_stats['layer_head_mean_tokens']
    num_layers = len(layer_head_tokens) if layer_head_tokens else 28
    num_heads = len(layer_head_tokens[0]) if layer_head_tokens else 28

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. average token per layer
    ax1 = axes[0, 0]
    layers = range(num_layers)
    ax1.bar(layers, dataset_stats['layer_mean_tokens'], color='steelblue')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Token Count')
    ax1.set_title(f'{dataset_name}: Average Token Count per Layer')
    ax1.set_xticks(layers)

    # 2. average token per head per layer (line plot, N lines for N heads)
    ax2 = axes[0, 1]
    layer_head_tokens = dataset_stats['layer_head_mean_tokens']  # N_layers x N_heads
    for head_idx in range(num_heads):
        head_tokens = [layer_head_tokens[layer_idx][head_idx] for layer_idx in range(num_layers)]
        ax2.plot(layers, head_tokens, marker='o', label=f'Head {head_idx}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Token Count')
    ax2.set_title(f'{dataset_name}: Average Token Count per Layer & Head')
    ax2.legend()
    ax2.set_xticks(layers)

    # 3. average token/length ratio per layer
    ax3 = axes[1, 0]
    ax3.bar(layers, dataset_stats['layer_data_ratios'], color='green')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Average Token/Length Ratio')
    ax3.set_title(f'{dataset_name}: Token/Length Ratio per Layer')
    ax3.set_xticks(layers)

    # 4. average token/length ratio per head per layer (line plot)
    ax4 = axes[1, 1]
    layer_head_ratios = dataset_stats['layer_head_mean_ratios']  # N_layers x N_heads
    for head_idx in range(num_heads):
        head_ratios = [layer_head_ratios[layer_idx][head_idx] for layer_idx in range(num_layers)]
        ax4.plot(layers, head_ratios, marker='o', label=f'Head {head_idx}')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Average Token/Length Ratio')
    ax4.set_title(f'{dataset_name}: Token/Length Ratio per Layer & Head')
    ax4.legend()
    ax4.set_xticks(layers)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{dataset_name}_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_data_ratio_distribution(dataset_stats, dataset_name, output_dir):
    """Create per-entry token/length distribution histogram and token count distribution histogram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. token/length ratio distribution
    ax1 = axes[0]
    ratios = dataset_stats['data_mean_ratios']
    ax1.hist(ratios, bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(ratios):.4f}')
    ax1.axvline(np.median(ratios), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(ratios):.4f}')
    ax1.set_xlabel('Token/Length Ratio')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{dataset_name}: Token/Length Ratio Distribution')
    ax1.legend()

    # 2. token count distribution
    ax2 = axes[1]
    tokens = [d['avg_token'] for d in dataset_stats['data_stats']]
    ax2.hist(tokens, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(tokens), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(tokens):.2f}')
    ax2.axvline(np.median(tokens), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(tokens):.2f}')
    ax2.set_xlabel('Token Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{dataset_name}: Token Count Distribution')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{dataset_name}_data_ratio_dist.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze TOPP file')
    parser.add_argument('--filename', type=str, default='qwen_topp32_09_0320_1')
    args = parser.parse_args()

    # auto-build paths
    input_file = os.path.join('record', f'{args.filename}.txt')
    output_dir = os.path.join('topp_analyze', f'analyze_{args.filename}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing file: {input_file}")
    datasets = parse_topp_file(input_file)

    print(f"Found {len(datasets)} datasets: {list(datasets.keys())}")

    # store stats for all datasets
    all_results = {}

    # analyze each dataset
    for dataset_name in datasets.keys():
        print(f"\nAnalyzing dataset: {dataset_name}")

        # create dataset output directory
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # analyze dataset
        stats = analyze_dataset(datasets, dataset_name)
        all_results[dataset_name] = stats

        # print statistics
        print(f"  Number of data: {stats['num_data']}")
        print(f"  Average length: {stats['total_length']:.2f}")
        print(f"  Average token: {stats['total_avg_token']:.2f}")
        print(f"  Average token/length: {stats['total_avg_ratio']:.4f}")

        # create visualizations
        print(f"  Creating visualizations...")

        # 1. layer x head heatmap (average token count)
        create_layer_head_heatmap(stats, dataset_name, dataset_dir)

        # 2. layer x head heatmap (token/length ratio)
        create_ratio_heatmap(stats, dataset_name, dataset_dir)

        # 3. summary statistics plots
        create_summary_plots(stats, dataset_name, dataset_dir)

        # 4. data distribution histograms
        create_data_ratio_distribution(stats, dataset_name, dataset_dir)

    # save summary statistics to JSON
    summary = {}
    for dataset_name, stats in all_results.items():
        summary[dataset_name] = {
            'num_data': stats['num_data'],
            'total_length': stats['total_length'],
            'total_avg_token': stats['total_avg_token'],
            'total_avg_ratio': stats['total_avg_ratio'],
            'layer_mean_tokens': stats['layer_mean_tokens'],
            'layer_head_mean_tokens': stats['layer_head_mean_tokens'],  # 28xN
            'layer_data_ratios': stats['layer_data_ratios'],
            'layer_head_mean_ratios': stats['layer_head_mean_ratios']  # 28xN
        }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
