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
    # 读取./dara/val_mapping.json
    with open('./data/val_mapping.json', 'r', encoding='utf-8') as f:
        val_mapping = json.load(f)
        # 把key中所有的image_2025替换为processed_images
        val_mapping = {k.replace('image_2025', 'processed_images'): v for k, v in val_mapping.items()}
        # 把\\替换为/
        val_mapping = {k.replace('\\', '/'): v for k, v in val_mapping.items()}
    # 循环
    model_mae_bins = {}  # 存储每个模型每个ATP区间的MAE列表
    for k_num in range(len(df_list)):
        df_name = df_list[k_num]
        print(f"正在处理文件: {df_name}")
        # 1. 加载预测结果
        df = pd.read_csv(f'./data/{df_name}')
        # 筛选在 val_mapping 中的样本
        df = df[df['image_path'].isin(val_mapping.keys())].reset_index(drop=True)
        # def extract_lt_number(path):
        #     match = re.search(r'LT(\d+)', path)
        #     return int(match.group(1)) if match else -1
        # def extract_plate_order(path):
        #     match = re.search(r'[Pp]late(\d+)', path)
        #     return int(match.group(1)) if match else 99  # 未匹配的放在最后
        # df['LT_num'] = df['image_path'].apply(extract_lt_number)
        # df['plate_order'] = df['image_path'].apply(extract_plate_order)
        # # 排序：先按 LT_num，再按 plate_order，保留其他顺序（稳定排序）
        # df = df.sort_values(by=['LT_num', 'plate_order'], kind='stable').reset_index(drop=True)
        # # 保存排序后的结果
        # df.to_csv('./data/pred_sorted.csv', index=False)



        # 3. 计算每个样本的误差指标
        df['MAE_diff'] = np.abs(df['pred_ATP'] - df['ATP'])
        # 3.1 绝对百分误差: |pred_ATP - ATP| / ATP * 100
        df['MAE_pct'] = np.abs(df['pred_ATP'] - df['ATP']) / df['ATP'] * 100



        # 6. 计算全体两两排序准确率
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

        # 7. 打印结果
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)

        print(f"mean_MAE_diff: {df['MAE_diff'].mean():.4f}")
        print(f"mean_MAE_pct: {df['MAE_pct'].mean():.4f}")
        print(f"overall_pairwise_ordering_accuracy: {overall_ordering_acc:.4f}")
        print("-"*50)

        # 输出MAPE
        mape = (np.abs(df['pred_ATP'] - df['ATP']) / (df['ATP'] + 1e-8)).mean() * 100
        print(f"📊 MAPE: {mape:.2f}%")

        # 输出相关系数
        # corr = df['pred_ATP'].corr(df['ATP'])
        corr = r2_score(df['pred_ATP'], df['ATP'])
        print(f"📊 r2: {corr:.4f}")

        # # ========================
        # # 分类预测正确和错误的 ATP MAE
        # # ========================
        correct_df = df[df['label'] == df['pred_label']]
        incorrect_df = df[df['label'] != df['pred_label']]
        conf_df = df[df['conf'] > 0.5]
        # 计算 MAE
        if not correct_df.empty:
            correct_mae = (correct_df['pred_ATP'] - correct_df['ATP']).abs().mean()
            print(f"✅ MAE (预测正确): {correct_mae:.4f}")
        else:
            print("⚠️ 没有预测正确的样本")
        if not incorrect_df.empty:
            incorrect_mae = (incorrect_df['pred_ATP'] - incorrect_df['ATP']).abs().mean()
            print(f"❌ MAE (预测错误): {incorrect_mae:.4f}")
        else:
            print("⚠️ 没有预测错误的样本")
        if not conf_df.empty:
            conf_mae = (conf_df['pred_ATP'] - conf_df['ATP']).abs().mean()
            print(f"✅ MAE (conf > 0.5): {conf_mae:.4f}")

        # 8. 计算 ATP 小于 500000 的样本的 MAE
        # 8. 将ATP划分为10段，并计算每段的MAE
        atp_values = df['ATP']
        min_atp, max_atp = atp_values.min(), atp_values.max()
        bin_edges2 = np.linspace(min_atp, max_atp + 1e-6, 11)  # 10段 => 11个边界

        mae_per_bin = []
        for i in range(10):
            bin_df = df[(df['ATP'] >= bin_edges2[i]) & (df['ATP'] < bin_edges2[i + 1])]
            if not bin_df.empty:
                mae = (bin_df['pred_ATP'] - bin_df['ATP']).abs().mean()
            else:
                mae = np.nan  # 没有样本
            mae_per_bin.append(mae)

        model_mae_bins[df_name] = mae_per_bin  # 保存当前模型的10段MAE

        print("-" * 50)



if __name__ == "__main__":
    main()
