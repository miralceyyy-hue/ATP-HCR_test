#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_model_metrics.py

This script evaluates the performance of the AD_cat_quant model on a set of predictions.
It reads './data/pred_updated.csv' (which must contain columns: image_path, label, pred_label, conf, ATP, pred_pct, pred_ATP),
reconstructs the bin ranges from the ATP mapping, and computes a range of metrics:
  1. Per-sample absolute percent error: |pred_ATP - ATP| / ATP * 100
  2. Per-sample percent error relative to the predicted label’s ATP range
  3. Per-sample percentile error: |pred_pct - actual_pct| * 100
  4. Per-label metrics:
       • sample count
       • classification accuracy
       • mean absolute percent error
       • mean range-normalized percent error
       • mean percentile error
       • pairwise ordering accuracy (only among correctly classified samples)
  5. Overall weighted averages of the above metrics (weighted by sample count per label)
  6. Overall pairwise ordering accuracy (among all samples)
All results are printed in a formatted table and summary.
"""

import pandas as pd
import numpy as np
import json
import itertools
import re
from sklearn.metrics import r2_score


def compute_bin_edges(mapping_dict, num_bins=8):
    """
    Compute bin edges of log10(ATP) values based on the original ATP mapping.

    Args:
        mapping_dict: dict, mapping from image path to ATP value
        num_bins: int, number of bins (default 8 bins, i.e., 9 edges)

    Returns:
        bin_edges: np.ndarray, array of length num_bins + 1
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)

    # Count samples in each bin
    bin_counts = np.histogram(log_atp_values, bins=bin_edges)[0]

    # Find the largest bin count and compute 10% of it
    max_bin_count = max(bin_counts)
    threshold = max_bin_count * 0.1

    # Accumulate bin counts until the threshold is exceeded
    cumulative = 0
    concat_bin_num = 0
    for count in bin_counts:
        cumulative += count
        concat_bin_num += 1
        if cumulative >= threshold:
            break
    return bin_edges, concat_bin_num


def main():
    df_list = ['pred_updated_AD_MutiHead_class12_44473515.csv','pred_updated_AD_MutiHead_class12.csv','pred_updated_googlenet_57797951.csv','pred_updated_inception_55565795.csv',
               'pred_updated_resnet_63536210.csv','pred_updated_vgg_64370553.csv',
               'pred_updated_class6_55300865.csv','pred_updated_class7_59445784.csv',
               'pred_updated_class8_55450629.csv','pred_updated_class9_58213844.csv',
               'pred_updated_class10_54893535.csv','pred_updated_class11_53686040.csv',
               'pred_updated_class12_49835097.csv','pred_updated_class13_58212806.csv',
               'pred_updated_class14_59482828.csv']
    bin_list = [12,12,12,12,12,12,6,7,8,9,10,11,12,13,14]
    df_list = ['pred_updated_resnet_MutiQ.csv','pred_updated_googlenet_MutiQ.csv',
               'pred_updated_inception_MutiQ.csv','pred_updated_vgg_MutiQ.csv',
               'pred_updated_AD_MutiHead_class6_53070808.csv', 'pred_updated_AD_MutiHead_class7_49840277.csv',
               'pred_updated_AD_MutiHead_class8_48337726.csv', 'pred_updated_AD_MutiHead_class9_49760909.csv',
               'pred_updated_AD_MutiHead_class10_46872753.csv', 'pred_updated_AD_MutiHead_class11_48507234.csv',
               'pred_updated_AD_MutiHead_class12_44097247.csv', 'pred_updated_AD_MutiHead_class13_46264555.csv',
               'pred_updated_AD_MutiHead_class14_45502575.csv']
    bin_list = [12,12,12,12, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # df_list = ['pred_updated_class12_49835097.csv']
    # bin_list = [12]
    # Read ./dara/val_mapping.json
    with open('./data/val_mapping.json', 'r', encoding='utf-8') as f:
        val_mapping = json.load(f)
        # Replace image_2025 with processed_images in every key
        val_mapping = {k.replace('image_2025', 'processed_images'): v for k, v in val_mapping.items()}
        # Replace backslashes with forward slashes
        val_mapping = {k.replace('\\', '/'): v for k, v in val_mapping.items()}
    model_mae_bins = {}  # Store the MAE list for each ATP interval per model

    for k_num in range(len(df_list)):
        df_name = df_list[k_num]
        print(f"Processing file: {df_name}")
        # 1. Load prediction results
        df = pd.read_csv(f'./data/{df_name}')
        num_bins = bin_list[k_num]
        # Keep samples that appear in val_mapping
        df = df[df['image_path'].isin(val_mapping.keys())].reset_index(drop=True)

        # def extract_lt_number(path):
        #     match = re.search(r'LT(\d+)', path)
        #     return int(match.group(1)) if match else -1
        # def extract_plate_order(path):
        #     match = re.search(r'[Pp]late(\d+)', path)
        #     return int(match.group(1)) if match else 99  # Unmatched values go last
        # # Extract the last 7 characters of the path (for lexicographic order)
        # def extract_last7(path):
        #     return str(path)[-7:]
        # # Add helper columns
        # df['LT_num'] = df['image_path'].apply(extract_lt_number)
        # df['plate_order'] = df['image_path'].apply(extract_plate_order)
        # df['last7'] = df['image_path'].apply(extract_last7)
        # # Sort: first by LT, then by plate, then by the lexicographic order of the last 7 characters
        # df = df.sort_values(by=['LT_num', 'plate_order', 'last7'], kind='stable').reset_index(drop=True)
        # # Save the sorted result
        # df.to_csv('./data/pred_sorted_AD_MutiHead_class12_44473515.csv', index=False)

        # 2. Load the full mapping and build label_ranges
        mapping_file = './data/processed_image_atp_mapping.json'
        with open(mapping_file, 'r', encoding='utf-8') as f:
            full_mapping = json.load(f)
        bin_edges, concat_bin_num = compute_bin_edges(full_mapping, num_bins=num_bins)
        # Build the log10(ATP) interval for each label
        label_ranges = {0: (bin_edges[0], bin_edges[concat_bin_num])}
        for i in range(concat_bin_num, len(bin_edges) - 1):
            label_ranges[i - concat_bin_num +1] = (bin_edges[i], bin_edges[i + 1])

        # Update the label column to match the actual ATP interval
        for i in range(len(df)):
            atp = df.loc[i, 'ATP']
            atp = np.log10(atp)
            # Find which interval the ATP value falls into
            for j in range(len(label_ranges)):
                low_log, high_log = label_ranges[j]
                if low_log <= atp < high_log:
                    df.loc[i,"label"] = j
                    break


        # 3. Compute error metrics for each sample
        df['MAE_diff'] = np.abs(df['pred_ATP'] - df['ATP'])
        # 3.1 Absolute percent error: |pred_ATP - ATP| / ATP * 100
        df['MAE_pct'] = np.abs(df['pred_ATP'] - df['ATP']) / df['ATP'] * 100



        # 3.2 Percent error relative to the predicted label range
        def compute_range_error(row):
            low_log, high_log = label_ranges[int(row['pred_label'])]
            range_atp = 10**high_log - 10**low_log
            return np.abs(row['pred_ATP'] - row['ATP']) / (range_atp + 1e-12) * 100
        df['range_MAE_pct'] = df.apply(compute_range_error, axis=1)

        # 3.3 Compute the actual percentile and percentile error
        def compute_actual_pct(row):
            # Calculate the percentile within the true label interval
            low_log, high_log = label_ranges[int(row['label'])]
            actual_log = np.log10(row['ATP'])
            pct = (actual_log - low_log) / (high_log - low_log + 1e-12)
            return np.clip(pct, 0.0, 1.0)
        df['actual_pct'] = df.apply(compute_actual_pct, axis=1)
        # 3.4 Percentile error: |pred_pct - actual_pct| * 100
        df['pct_error'] = np.abs(df['pred_pct'] - df['actual_pct']) * 100

        # 4. Compute metrics for each label
        per_label = []
        for label, grp in df.groupby('label'):
            count = len(grp)
            cls_acc = (grp['pred_label'] == grp['label']).mean()
            # Use correctly classified samples for downstream regression metrics (currently using all samples)
            # correct = grp[grp['pred_label'] == label].reset_index(drop=True)
            correct = grp.reset_index(drop=True)
            mae_diff = correct['MAE_diff'].mean() if len(correct)>0 else np.nan
            mae_pct = correct['MAE_pct'].mean() if len(correct)>0 else np.nan
            range_mae_pct = correct['range_MAE_pct'].mean() if len(correct)>0 else np.nan
            pct_mae = correct['pct_error'].mean() if len(correct)>0 else np.nan

            # Compute pairwise ordering accuracy for the selected samples
            n_corr = len(correct)
            if n_corr < 2:
                ordering_acc = np.nan
            else:
                total, corr_pairs = 0, 0
                sig_total, sig_corr_pairs = 0, 0

                # Get the interval for this label
                low_log, high_log = label_ranges[label]
                range_threshold = (10 ** high_log - 10 ** low_log) * 0.1

                for i, j in itertools.combinations(range(n_corr), 2):
                    actual_diff = correct.loc[i, 'ATP'] - correct.loc[j, 'ATP']
                    pred_diff   = correct.loc[i, 'pred_ATP'] - correct.loc[j, 'pred_ATP']
                    if actual_diff * pred_diff > 0:
                        corr_pairs += 1
                    total += 1

                    if abs(actual_diff) > range_threshold:
                        sig_total += 1
                        if actual_diff * pred_diff > 0:
                            sig_corr_pairs += 1

                ordering_acc = corr_pairs / total
                significant_ordering_acc = sig_corr_pairs / sig_total if sig_total > 0 else np.nan


            per_label.append({
                'label': label,
                'count': count,
                'classification_accuracy': cls_acc,
                'mean_MAE_diff': mae_diff,
                'mean_MAE_pct': mae_pct,
                'mean_range_MAE_pct': range_mae_pct,
                'mean_pct_error': pct_mae,
                'pairwise_ordering_acc': ordering_acc,
                'significant_pairwise_ordering_acc': significant_ordering_acc

            })
        metrics_df = pd.DataFrame(per_label).set_index('label')

        # 5. Compute overall weighted metrics
        total_samples = metrics_df['count'].sum()
        weighted = {}
        for col in ['classification_accuracy','mean_MAE_diff', 'mean_MAE_pct', 'mean_range_MAE_pct', 'mean_pct_error', 'pairwise_ordering_acc','significant_pairwise_ordering_acc']:
            weighted[col] = (metrics_df[col] * metrics_df['count']).sum() / total_samples

        # 6. Compute pairwise ordering accuracy across all samples
        # df = df[df['pred_label'] == df['label']].reset_index(drop=True)
        n_all = len(df)
        if n_all < 2:
            overall_ordering_acc = np.nan
        else:
            total_pairs, corr_pairs = 0, 0
            for i, j in itertools.combinations(range(n_all), 2):
                actual_diff = df.loc[i, 'ATP'] - df.loc[j, 'ATP']
                pred_diff   = df.loc[i, 'pred_ATP'] - df.loc[j, 'pred_ATP']
                if actual_diff * pred_diff > 0:
                    corr_pairs += 1
                total_pairs += 1
            overall_ordering_acc = corr_pairs / total_pairs

        # 7. Print results
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        print("\n=== Per-Label Metrics ===")
        print(metrics_df.to_string(), "\n")

        print(f"=== Overall Weighted Metrics in {df_name} ===")
        for name, val in weighted.items():
            print(f"{name}: {val:.4f}")
        print(f"overall_pairwise_ordering_accuracy: {overall_ordering_acc:.4f}")

        # Output MAPE
        mape = (np.abs(df['pred_ATP'] - df['ATP']) / (df['ATP'] + 1e-8)).mean() * 100
        print(f"📊 MAPE: {mape:.2f}%")

        # Output correlation coefficient
        # corr = df['pred_ATP'].corr(df['ATP'])
        corr = r2_score(df['pred_ATP'], df['ATP'])
        print(f"📊 r2: {corr:.4f}")


        # # ========================
        # # ATP MAE for correct and incorrect predictions
        # # ========================
        correct_df = df[df['label'] == df['pred_label']]
        incorrect_df = df[df['label'] != df['pred_label']]
        conf_df = df[df['conf'] > 0.5]
        # Compute MAE
        if not correct_df.empty:
            correct_mae = (correct_df['pred_ATP'] - correct_df['ATP']).abs().mean()
            print(f"✅ MAE (predicted correctly): {correct_mae:.4f}")
        else:
            print("⚠️ No correctly predicted samples")
        if not incorrect_df.empty:
            incorrect_mae = (incorrect_df['pred_ATP'] - incorrect_df['ATP']).abs().mean()
            print(f"❌ MAE (预测错误): {incorrect_mae:.4f}")
        else:
            print("⚠️ predicted incorrectl")
        if not conf_df.empty:
            conf_mae = (conf_df['pred_ATP'] - conf_df['ATP']).abs().mean()
            print(f"✅ MAE (conf > 0.5): {conf_mae:.4f}")


        # 8. Compute MAE for samples below 500000 ATP
        # 8. Split ATP into 10 intervals and compute each interval's MAE
        atp_values = df['ATP']
        min_atp, max_atp = atp_values.min(), atp_values.max()
        bin_edges2 = np.linspace(min_atp, max_atp + 1e-6, 11)  # 10 intervals => 11 edges

        mae_per_bin = []
        for i in range(10):
            bin_df = df[(df['ATP'] >= bin_edges2[i]) & (df['ATP'] < bin_edges2[i + 1])]
            if not bin_df.empty:
                mae = (bin_df['pred_ATP'] - bin_df['ATP']).abs().mean()
            else:
                mae = np.nan  # No samples
            mae_per_bin.append(mae)

        model_mae_bins[df_name] = mae_per_bin  # Save the 10-bin MAE for the current model

        print("-" * 50)

    # Build interval labels
    bin_labels = ["{:.2e}-{:.2e}".format(bin_edges[i], bin_edges[i + 1]) for i in range(10)]
    # Create DataFrame
    mae_df = pd.DataFrame.from_dict(model_mae_bins, orient='index', columns=bin_labels)
    # Print MAE DataFrame
    print("\n=== MAE per ATP bin (rows: models, columns: ATP bins) ===")
    print(mae_df)


if __name__ == "__main__":
    main()
