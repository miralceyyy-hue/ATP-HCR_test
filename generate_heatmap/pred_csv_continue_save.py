import torch
import pandas as pd
import json
import numpy as np
import cv2
from sympy.integrals.meijerint_doc import category
from torchvision import transforms
from model.AD2DMIT import AD2D_MIL_bin_Q,AD2D_MIL_Log_Continue
from PIL import Image
from tqdm import tqdm
import itertools
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


# 1. Load mapping and label_ranges
mapping_file = './data/processed_image_atp_mapping.json'
num_bins = 14  # number of bins
allowed_prefixes = None
# allowed_prefixes = ['LT19']  # only handle these prefixes
with open(mapping_file, 'r', encoding='utf-8') as f:
    full_mapping = json.load(f)

# 2. Define image transform
transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
    ])

# 3. Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AD2D_MIL_Log_Continue(
        in_channel=1,  # RGB input
        hidden=256,  # hidden dimension
        image_size=512,
        patches=64
    ).to(device)
result_name = "AD_continue_best_class.pth"

# Extract trailing number in filename
result_num = "AD_continue"
model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
model.to(device)
model.eval()

# ============ Main workflow ============
df = pd.read_csv('./data/pred.csv')
# df = pd.read_csv('./data/image_atp_mapping_LT5.csv')

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

    # Save original image to ./plts/heatmap/original_images/
    original_save_path = f"./plts/heatmap/original_images/{level}/{image_id}.png"
    os.makedirs(os.path.dirname(original_save_path), exist_ok=True)
    if not os.path.exists(original_save_path):
        try:
            original_img = Image.open(real_path).convert('RGB')
            original_img.save(original_save_path)
        except:
            print("continue:", real_path)
            continue

    try:
        img = cv2.imread(real_path,0)
    except:
        print("continue:",real_path)
        continue
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

    save_path = f"./plts/heatmap/{result_name}/{level}/{image_id}.png"
    visualize_heatmap_overlay(input_tensor[0], att_map, save_path)
    sampled_per_level[level].append(image_id)

print("✅ Visualization finished and saved to ./plts/heatmap/")