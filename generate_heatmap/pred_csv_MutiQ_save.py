import torch
import pandas as pd
import json
import numpy as np
import cv2
from torchvision import transforms
from model.AD2DMIT import AD2D_MIL_bin_MutiQ
from PIL import Image
from tqdm import tqdm
import itertools
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============ 可视化函数 ============
def visualize_heatmap_overlay(image_tensor, heatmap, save_path):
    image_np = image_tensor.squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    # ax.imshow(image_np, cmap='gray')
    sns.heatmap(heatmap, cmap='viridis', alpha=1, ax=ax, cbar=False)
    ax.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def get_log10_level(atp):
    return f"1e{int(np.floor(np.log10(atp)))}"

# 计算bin_edge
def compute_bin_edges(mapping_dict, num_bins=8):
    """
    根据原始ATP映射，计算log10(ATP)后的分箱边界。

    参数:
        mapping_dict: dict，图像路径到ATP值的映射
        num_bins: int，分箱数量（默认8个箱，对应9个边界）

    返回:
        bin_edges: np.ndarray，长度为 num_bins+1 的数组
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)

    # 统计每个 bin 中的样本数
    bin_counts = np.histogram(log_atp_values, bins=bin_edges)[0]

    # 找出最大 bin 样本数，并计算其10%
    max_bin_count = max(bin_counts)
    threshold = max_bin_count * 0.1

    # 累加前几个 bin 的样本数，直到超过阈值
    cumulative = 0
    concat_bin_num = 0
    for count in bin_counts:
        cumulative += count
        concat_bin_num += 1
        if cumulative >= threshold:
            break
    return bin_edges, concat_bin_num

# 1. 加载 mapping 和 label_ranges
mapping_file = './data/processed_image_atp_mapping.json'
# df_list = ['pred_updated_resnet_MutiQ.csv', 'pred_updated_googlenet_MutiQ.csv',
#            'pred_updated_inception_MutiQ.csv', 'pred_updated_vgg_MutiQ.csv',
#            'pred_updated_AD_MutiHead_class6_53070808.csv', 'pred_updated_AD_MutiHead_class7_49840277.csv',
#            'pred_updated_AD_MutiHead_class8_48337726.csv', 'pred_updated_AD_MutiHead_class9_49760909.csv',
#            'pred_updated_AD_MutiHead_class10_46872753.csv', 'pred_updated_AD_MutiHead_class11_48507234.csv',
#            'pred_updated_AD_MutiHead_class12_44097247.csv', 'pred_updated_AD_MutiHead_class13_46264555.csv',
#            'pred_updated_AD_MutiHead_class14_45502575.csv']
# result_name_list = [
#                      'AD_MutiHead_class6_best_53070808.pth', 'AD_MutiHead_class7_best_49840277.pth',
#                      'AD_MutiHead_class8_best_48337726.pth', 'AD_MutiHead_class9_best_49760909.pth',
#                      'AD_MutiHead_class10_best_46872753.pth', 'AD_MutiHead_class11_best_48507234.pth',
#                      'AD_MutiHead_class12_best_44097247.pth', 'AD_MutiHead_class13_best_46264555.pth',
#                      'AD_MutiHead_class14_best_45502575.pth']
# bin_list = [ 6, 7, 8, 9, 10, 11, 12, 13, 14]
# category_list = [4 ,5 ,6 ,7 ,8 ,8 ,9 ,10 ,11]
result_name_list = ['AD_MutiHead_class12_best_44097247.pth']
bin_list = [12]
category_list = [9]
allowed_prefixes = None
# allowed_prefixes = ['LT19']  # 只处理这两类图像

for result_name, num_bins, category in zip(result_name_list, bin_list, category_list):
    print(f"Processing model: {result_name}, num_bins: {num_bins}, category: {category}")
    # num_bins = 12  # 分箱数量
    # category = 9
    # result_name = "AD_MutiHead_class12_best_44097247.pth"
    # # 6对应4分类、7对应5分类、8对应6分类、9对应7分类、10对应8分类、11对应8分类、12对应9分类、13对应10分类、14对应11分类

    with open(mapping_file, 'r', encoding='utf-8') as f:
        full_mapping = json.load(f)
    bin_edges, concat_bin_num = compute_bin_edges(full_mapping, num_bins=num_bins)
    # 计算 label_ranges（类别 -> log10(ATP) 区间）
    label_ranges = {0: (bin_edges[0], bin_edges[concat_bin_num])}
    for i in range(concat_bin_num, len(bin_edges) - 1):
        label_ranges[i - concat_bin_num+1] = (bin_edges[i], bin_edges[i + 1])

    # 2. 定义图像 transform
    transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49188775],
                                 std=[0.26558745])
        ])

    # 3. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AD2D_MIL_bin_MutiQ(
            in_channel=1,  # RGB输入
            hidden=256,  # 隐藏层维度
            category=category,
            image_size=512,
            patches=64,
            label_ranges=label_ranges,
        ).to(device)

    # 获取名字中在最后的数字
    result_num = result_name.split('_')[-1].split('.')[0]
    model_name = "AD_MutiHead_class"+str(num_bins)+"_" + result_num

    model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
    model.to(device)
    model.eval()

    # ============ 主流程 ============
    df = pd.read_csv('./data/pred.csv')
    # df = pd.read_csv('./data/image_atp_mapping_LT5.csv')
    sampled_per_level = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Heatmap generating"):
        img_path = row['image_path'].replace("\\", "/")  # 防止路径分隔符不同
        filename = os.path.basename(img_path)  # 取文件名
        project_id = filename.split('_')[0]  # 例如 LT19、LT5、LT18 等
        # project_id = project_id.split('LT')[1]
        # 如果不在允许列表中，跳过
        if allowed_prefixes is not None:
            if project_id not in allowed_prefixes:
                continue

        atp = row['ATP']
        level = get_log10_level(atp)
        sampled_per_level.setdefault(level, [])
        if len(sampled_per_level[level]) >= 500:
            continue

        img_path = row['image_path']
        real_path = img_path.replace("image_2025", "processed_images")
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(real_path, 0)
        if img is None:
            continue
        pil_img = Image.fromarray(img)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        # print(model.feature_extractor)
        with torch.no_grad():
            features = model.feature_extractor[:4](input_tensor)  # (B, C, H, W)
            B, C, H, W = features.shape
            # out = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
            # # weights = model.attention(out).view(1, -1, 1)  # (1, N, 1)
            # out = out.view(1, -1, C)
            att_map = features.mean(dim=1, keepdim=True).squeeze(0).squeeze(0).cpu()  # (H, W)

        save_path = "original_images"
        save_path = f"./plts/heatmap/{model_name}/{level}/{image_id}.png"
        visualize_heatmap_overlay(input_tensor[0], att_map, save_path)
        sampled_per_level[level].append(image_id)

    print("✅ 可视化完成并保存到 ./plts/heatmap/")