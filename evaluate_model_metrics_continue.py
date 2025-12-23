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




def main():
    df_list = ['pred_binary_model_result_DAC2.csv','pred_binary_model_result_AD2DMIT.csv','pred_updated_AD_continue.csv']
    # Read ./dara/val_mapping.json
    with open('./data/val_mapping.json', 'r', encoding='utf-8') as f:
        val_mapping = json.load(f)
        # Replace image_2025 with processed_images in every key
        val_mapping = {k.replace('image_2025', 'processed_images'): v for k, v in val_mapping.items()}
        # Replace \\ with /
        val_mapping = {k.replace('\\', '/'): v for k, v in val_mapping.items()}
    # Loop
    model_mae_bins = {}  # Store MAEs per ATP interval for each model
    for k_num in range(len(df_list)):
        df_name = df_list[k_num]
        print(f"Processing file: {df_name}")
        # 1. Load prediction results
        df = pd.read_csv(f'./data/{df_name}')
        # Filter samples present in val_mapping
        df = df[df['image_path'].isin(val_mapping.keys())].reset_index(drop=True)
        # def extract_lt_number(path):
        #     match = re.search(r'LT(\d+)', path)
        #     return int(match.group(1)) if match else -1
        # def extract_plate_order(path):
        #     match = re.search(r'[Pp]late(\d+)', path)
        #     return int(match.group(1)) if match else 99  # Put unmatched at the end
        # df['LT_num'] = df['image_path'].apply(extract_lt_number)
        # df['plate_order'] = df['image_path'].apply(extract_plate_order)
        # # Sort by LT_num then plate_order while keeping stability
        # df = df.sort_values(by=['LT_num', 'plate_order'], kind='stable').reset_index(drop=True)
        # # Save the sorted results
        # df.to_csv('./data/pred_sorted.csv', index=False)



        # 3. Compute per-sample error metrics
        df['MAE_diff'] = np.abs(df['pred_ATP'] - df['ATP'])
        # 3.1 Absolute percent error: |pred_ATP - ATP| / ATP * 100
        df['MAE_pct'] = np.abs(df['pred_ATP'] - df['ATP']) / df['ATP'] * 100



        # 6. Compute overall pairwise ordering accuracy
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

        print(f"mean_MAE_diff: {df['MAE_diff'].mean():.4f}")
        print(f"mean_MAE_pct: {df['MAE_pct'].mean():.4f}")
        print(f"overall_pairwise_ordering_accuracy: {overall_ordering_acc:.4f}")
        print("-"*50)

        # Output MAPE
        mape = (np.abs(df['pred_ATP'] - df['ATP']) / (df['ATP'] + 1e-8)).mean() * 100
        print(f"📊 MAPE: {mape:.2f}%")

        # Output R^2
        # corr = df['pred_ATP'].corr(df['ATP'])
        corr = r2_score(df['pred_ATP'], df['ATP'])
        print(f"📊 r2: {corr:.4f}")

        # # ========================
        # # ATP MAE for correct vs incorrect classifications
        # # ========================
        correct_df = df[df['label'] == df['pred_label']]
        incorrect_df = df[df['label'] != df['pred_label']]
        conf_df = df[df['conf'] > 0.5]
        # Compute MAE
        if not correct_df.empty:
            correct_mae = (correct_df['pred_ATP'] - correct_df['ATP']).abs().mean()
            print(f"✅ MAE (correct predictions): {correct_mae:.4f}")
        else:
            print("⚠️ No correctly predicted samples")
        if not incorrect_df.empty:
            incorrect_mae = (incorrect_df['pred_ATP'] - incorrect_df['ATP']).abs().mean()
            print(f"❌ MAE (incorrect predictions): {incorrect_mae:.4f}")
        else:
            print("⚠️ No incorrectly predicted samples")
        if not conf_df.empty:
            conf_mae = (conf_df['pred_ATP'] - conf_df['ATP']).abs().mean()
            print(f"✅ MAE (conf > 0.5): {conf_mae:.4f}")

        # 8. Compute MAE for ATP < 500000
        # 8. Split ATP into 10 bins and compute MAE per bin
        atp_values = df['ATP']
        min_atp, max_atp = atp_values.min(), atp_values.max()
        bin_edges2 = np.linspace(min_atp, max_atp + 1e-6, 11)  # 10 bins => 11 edges

        mae_per_bin = []
        for i in range(10):
            bin_df = df[(df['ATP'] >= bin_edges2[i]) & (df['ATP'] < bin_edges2[i + 1])]
            if not bin_df.empty:
                mae = (bin_df['pred_ATP'] - bin_df['ATP']).abs().mean()
            else:
                mae = np.nan  # No samples
            mae_per_bin.append(mae)

        model_mae_bins[df_name] = mae_per_bin  # Save 10-bin MAE for the current model

        print("-" * 50)



if __name__ == "__main__":
    main()
