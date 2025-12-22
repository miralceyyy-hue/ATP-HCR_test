import torch
import pandas as pd
import json
import numpy as np
import cv2
from sympy.integrals.meijerint_doc import category
from torchvision import transforms
from model.ATP_HCR import ATP_HCR_bin_Q,ATP_HCR_Log_Continue
from PIL import Image
from tqdm import tqdm
import itertools



# 1. 加载 mapping 和 label_ranges
mapping_file = './data/processed_image_atp_mapping.json'
num_bins = 14  # 分箱数量
with open(mapping_file, 'r', encoding='utf-8') as f:
    full_mapping = json.load(f)

# 2. 定义图像 transform
transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
    ])

# 3. 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ATP_HCR_Log_Continue(
        in_channel=1,  # RGB输入
        hidden=256,  # 隐藏层维度
        image_size=512,
        patches=64
    ).to(device)
result_name = "AD_continue_best_class.pth"

# 获取名字中在最后的数字
result_num = "AD_continue"
model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
model.to(device)
model.eval()

# 4. 加载 pred.csv
df = pd.read_csv('./data/pred.csv')

pred_labels = []
pred_pcts = []
pred_atps = []

for path in tqdm(df['image_path'], desc="Predicting"):
    # 处理路径
    real_path = path.replace("image_2025", "processed_images")
    img = cv2.imread(real_path, 0)
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {real_path}")
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        logits= model(img)
        log_atp_pred = logits.item()  # 直接输出 log10(ATP)
        atp_pred = 10 ** log_atp_pred

    # 记录
    pred_atps.append(atp_pred)

# 5. 保存新 CSV
df['pred_ATP'] = pred_atps
df.to_csv('./data/pred_updated_'+result_num+'.csv', index=False)
print("✅ 保存为 pred_updated.csv 成功")

# 加一列百分误差
# 读取更新后的CSV文件
df = pd.read_csv('./data/pred_updated_'+result_num+'.csv')
# 计算百分误差（百分比）
df['ATP_pct_error'] = (abs(df['pred_ATP'] - df['ATP']) / df['ATP']) * 100
# 打印平均ATP百分误差
# 只计算label==pred_label列的均值
print(df.shape)

# 可选：保存带误差列的新CSV
# df.to_csv('./data/pred_updated.csv', index=False)

# 计算两两相对大小正确率
results = {}
# 只保留 label == pred_label 的行
df_correct = df
for lab, group in df_correct.groupby('label'):
    ats = group['ATP'].values
    preds = group['pred_ATP'].values
    n = len(ats)
    # 如果该类别样本少于 2 个，则无法成对比较，跳过或记为 NaN
    if n < 2:
        results[lab] = float('nan')
        continue
    total_pairs = 0
    correct_pairs = 0
    # 两两配对
    for i, j in itertools.combinations(range(n), 2):
        actual_diff = ats[i] - ats[j]
        pred_diff = preds[i] - preds[j]

        # 只要二者符号相同（>0/<0），就算预测对；忽略相等的情况
        if actual_diff * pred_diff > 0:
            correct_pairs += 1
        total_pairs += 1

    accuracy = correct_pairs / total_pairs
    results[lab] = accuracy
# 打印每个 label 的排序准确率
for lab, acc in results.items():
    print(f"Label {lab}: pairwise ordering accuracy = {acc:.3f}")
# （可选）总体平均
valid_accs = [v for v in results.values() if not pd.isna(v)]
overall_acc = sum(valid_accs) / len(valid_accs)
print(f"\nOverall pairwise ordering accuracy (mean across labels): {overall_acc:.3f}")