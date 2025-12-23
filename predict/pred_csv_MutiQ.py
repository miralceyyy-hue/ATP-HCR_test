import torch
import pandas as pd
import json
import numpy as np
import cv2
from sympy.integrals.meijerint_doc import category
from torchvision import transforms
from model.ATP_HCR import ATP_HCR_bin_MutiQ
from PIL import Image
from tqdm import tqdm
import itertools
import torch.nn.functional as F

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
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
    ])

# 3. Load model
category = 9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ATP_HCR_bin_MutiQ(
        in_channel=1,  # RGB input
        hidden=256,  # hidden dimension
        category=category,
        image_size=512,
        patches=64,
        label_ranges=label_ranges,
    ).to(device)
result_name = "AD_MutiHead_class12_best_44097247.pth"
# For checkpoints: 6→4 classes, 7→5 classes, 8→6 classes, 9→7 classes, 10→8 classes, 11→8 classes, 12→9 classes, 13→10 classes, 14→11 classes

# Extract the trailing number in the filename
result_num = result_name.split('_')[-1]
result_num = result_num.split('.')[0]
result_num = "AD_MutiHead_class12_"+result_num
model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
model.to(device)
model.eval()

# 4. Load pred.csv
df = pd.read_csv('./data/pred.csv')

pred_labels = []
pred_pcts = []
pred_atps = []
pred_max_prob_atps = []
bin_probs_lists = []

for path in tqdm(df['image_path'], desc="Predicting"):
    # Normalize path
    real_path = path.replace("image_2025", "processed_images")
    img = cv2.imread(real_path, 0)
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {real_path}")
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        # logits, pct_pred = model(img)
        atp_pred, logits, pcts, atp_max_prob_pred = model(img)
        bin_probs = F.softmax(logits, dim=1)
        atp_pred = atp_pred.view(-1)
        atp_max_prob_pred = atp_max_prob_pred.view(-1)
        cls_pred = torch.argmax(bin_probs, dim=1).item()
        preds = bin_probs.argmax(dim=1)
        pct_pred = pcts[torch.arange(pcts.size(0)), preds]
        pct = pct_pred.item()
        low, high = label_ranges[cls_pred]
        log_atp_pred = low + pct * (high - low)
        # atp_pred = 10 ** log_atp_pred

    # Record predictions
    pred_labels.append(cls_pred)
    pred_pcts.append(pct)
    pred_atps.append(atp_pred.item())
    pred_max_prob_atps.append(atp_max_prob_pred.item())
    bin_probs_lists.append(bin_probs.cpu().numpy())

# 5. Save new CSV
df['pred_label'] = pred_labels
df['pred_pct'] = pred_pcts
df['pred_ATP'] = pred_atps
df['pred_max_prob_ATP'] = pred_max_prob_atps
df['bin_probs'] = bin_probs_lists
df.to_csv('./data/pred_updated_'+result_num+'.csv', index=False)
print("✅ Saved as pred_updated.csv")

# Add a column for percentage error
# Read the updated CSV
df = pd.read_csv('./data/pred_updated_'+result_num+'.csv')
# Compute percentage error
df['ATP_pct_error'] = (abs(df['pred_ATP'] - df['ATP']) / df['ATP']) * 100
# Print mean ATP percentage error for rows with correct class prediction
print(df.shape)
df2 = df[df['label'] == df['pred_label']]
print(df2.shape)
mean_error = df2['ATP_pct_error'].mean()
print(f"📊 Mean ATP percentage error: {mean_error:.2f}%")
# Optional: save the CSV with the error column
# df.to_csv('./data/pred_updated.csv', index=False)

# Compute pairwise ordering accuracy
results = {}
# Keep only rows where the predicted class matches the label
df_correct = df[df['label'] == df['pred_label']]
for lab, group in df_correct.groupby('label'):
    ats = group['ATP'].values
    preds = group['pred_ATP'].values
    n = len(ats)
    # Skip or mark NaN when a class has fewer than two samples
    if n < 2:
        results[lab] = float('nan')
        continue
    total_pairs = 0
    correct_pairs = 0
    # Pairwise comparison
    for i, j in itertools.combinations(range(n), 2):
        actual_diff = ats[i] - ats[j]
        pred_diff = preds[i] - preds[j]

        # Count correct if the signs match (>0/<0); ignore equality
        if actual_diff * pred_diff > 0:
            correct_pairs += 1
        total_pairs += 1

    accuracy = correct_pairs / total_pairs
    results[lab] = accuracy
# Print pairwise accuracy per label
for lab, acc in results.items():
    print(f"Label {lab}: pairwise ordering accuracy = {acc:.3f}")
# (Optional) Overall average
valid_accs = [v for v in results.values() if not pd.isna(v)]
overall_acc = sum(valid_accs) / len(valid_accs)
print(f"\nOverall pairwise ordering accuracy (mean across labels): {overall_acc:.3f}")