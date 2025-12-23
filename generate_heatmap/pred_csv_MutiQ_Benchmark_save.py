import torch
import pandas as pd
import json
import numpy as np
import cv2
from torchvision import transforms
from model.Benchmarks import BinaryBenchMarkNet_MutiQ
from PIL import Image
from tqdm import tqdm
import itertools
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============ Visualization helper ============
def visualize_heatmap_overlay(image_tensor, heatmap, save_path):
    image_np = image_tensor.squeeze().cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))  # reshape to (H, W, 3)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    # ax.imshow(image_np)
    sns.heatmap(heatmap, cmap='viridis', alpha=1, ax=ax, cbar=False)
    ax.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def get_log10_level(atp):
    return f"1e{int(np.floor(np.log10(atp)))}"

# Compute bin edges in log10(ATP) space
def compute_bin_edges(mapping_dict, num_bins=8):
    """
    Calculate log10(ATP) bin boundaries from the original ATP mapping.

    Args:
        mapping_dict: dict, mapping from image path to ATP value
        num_bins: int, number of bins (defaults to 8 bins, i.e., 9 edges)

    Returns:
        bin_edges: np.ndarray with length num_bins + 1
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)

    # Count samples per bin
    bin_counts = np.histogram(log_atp_values, bins=bin_edges)[0]

    # Find the largest bin size and compute 10% of it
    max_bin_count = max(bin_counts)
    threshold = max_bin_count * 0.1

    # Accumulate leading bins until the threshold is reached
    cumulative = 0
    concat_bin_num = 0
    for count in bin_counts:
        cumulative += count
        concat_bin_num += 1
        if cumulative >= threshold:
            break
    return bin_edges, concat_bin_num

# 1. Load mapping and label_ranges
mapping_file = './data/processed_image_atp_mapping.json'
num_bins = 12  # number of bins
category = 9
backbone = 'vgg'
result_name = "vgg_MutiHead_best.pth"
allowed_prefixes = None
# allowed_prefixes = ['LT19']  # only handle these prefixes
# vgg feature 16 \resnet feature 5 \googlenet feature 4 \inception feature 4

with open(mapping_file, 'r', encoding='utf-8') as f:
    full_mapping = json.load(f)
bin_edges, concat_bin_num = compute_bin_edges(full_mapping, num_bins=num_bins)
# Calculate label_ranges (class -> log10(ATP) interval)
label_ranges = {0: (bin_edges[0], bin_edges[concat_bin_num])}
for i in range(concat_bin_num, len(bin_edges) - 1):
    label_ranges[i - concat_bin_num+1] = (bin_edges[i], bin_edges[i + 1])

# 2. Define image transform
transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775]*3,
                             std=[0.26558745]*3)
    ])

# 3. Load model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BinaryBenchMarkNet_MutiQ(
    backbone=backbone,
    category=category,
    label_ranges=label_ranges,
).to(device)
# For checkpoints: 6→4 classes, 7→5 classes, 8→6 classes, 9→7 classes, 10→8 classes, 11→8 classes, 12→9 classes, 13→10 classes, 14→11 classes

# Extract trailing number in filename
# result_num = result_name.split('_')[-1]
# result_num = result_num.split('.')[0]
result_num = "MutiQ"
model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
model.to(device)
model.eval()
# Remove avgpool and fc layers for googlenet/inception backbones
model.feature_extractor.feature_extractor = torch.nn.Sequential(
    *list(model.feature_extractor.feature_extractor.children())[:-3]
)

# ============ Main workflow ============
df = pd.read_csv('./data/pred.csv')

sampled_per_level = {}

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Heatmap generating"):
    img_path = row['image_path'].replace("\\", "/")  # normalize separators
    filename = os.path.basename(img_path)  # get filename
    project_id = filename.split('_')[0]  # e.g., LT19, LT5, LT18
    # project_id = project_id.split('LT')[1]
    # Skip if not in allowlist
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

    img = cv2.imread(real_path)
    if img is None:
        continue
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    # print(model.feature_extractor.feature_extractor)

    with torch.no_grad():
        features = model.feature_extractor.feature_extractor[:16](input_tensor)  # (B, C, H, W)
        B, C, H, W = features.shape
        # out = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # # weights = model.attention(out).view(1, -1, 1)  # (1, N, 1)
        # out = out.view(1, -1, C)
        att_map = features.mean(dim=1, keepdim=True).squeeze(0).squeeze(0).cpu()  # (H, W)

    save_path = f"./plts/heatmap/{result_name}/{level}/{image_id}.png"
    visualize_heatmap_overlay(input_tensor[0], att_map, save_path)
    sampled_per_level[level].append(image_id)

print("✅ Visualization finished and saved to ./plts/heatmap/")