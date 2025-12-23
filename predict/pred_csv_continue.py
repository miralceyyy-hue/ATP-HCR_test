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



# 1. Load mapping and label_ranges
mapping_file = './data/processed_image_atp_mapping.json'
num_bins = 14  # Number of bins
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
model = ATP_HCR_Log_Continue(
        in_channel=1,  # RGB input
        hidden=256,  # Hidden layer dimension
        image_size=512,
        patches=64
    ).to(device)
result_name = "AD_continue_best_class.pth"

# Extract the trailing number in the filename
result_num = "AD_continue"
model.load_state_dict(torch.load('./trained_models/'+result_name, map_location=device))
model.to(device)
model.eval()

# 4. Load pred.csv
df = pd.read_csv('./data/pred.csv')

pred_labels = []
pred_pcts = []
pred_atps = []

for path in tqdm(df['image_path'], desc="Predicting"):
    # Normalize path
    real_path = path.replace("image_2025", "processed_images")
    img = cv2.imread(real_path, 0)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {real_path}")
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits= model(img)
        log_atp_pred = logits.item()  # Directly outputs log10(ATP)
        atp_pred = 10 ** log_atp_pred

    # Record prediction
    pred_atps.append(atp_pred)

# 5. Save new CSV
df['pred_ATP'] = pred_atps
df.to_csv('./data/pred_updated_'+result_num+'.csv', index=False)
print("✅ Saved pred_updated.csv successfully")

# Add percent error column
# Reload the updated CSV
df = pd.read_csv('./data/pred_updated_'+result_num+'.csv')
# Compute percentage error
df['ATP_pct_error'] = (abs(df['pred_ATP'] - df['ATP']) / df['ATP']) * 100
# Print mean ATP percentage error
# Only average rows where label == pred_label
print(df.shape)

# Optional: save CSV with the error column
# df.to_csv('./data/pred_updated.csv', index=False)

# Compute pairwise ordering accuracy
results = {}
# Keep only rows where label == pred_label
df_correct = df
for lab, group in df_correct.groupby('label'):
    ats = group['ATP'].values
    preds = group['pred_ATP'].values
    n = len(ats)
    # If a class has fewer than 2 samples, skip or mark NaN
    if n < 2:
        results[lab] = float('nan')
        continue
    total_pairs = 0
    correct_pairs = 0
    # Pairwise comparison
    for i, j in itertools.combinations(range(n), 2):
        actual_diff = ats[i] - ats[j]
        pred_diff = preds[i] - preds[j]

        # Prediction is correct if signs match (>0/<0); ignore equals
        if actual_diff * pred_diff > 0:
            correct_pairs += 1
        total_pairs += 1

    accuracy = correct_pairs / total_pairs
    results[lab] = accuracy
# Print ordering accuracy per label
for lab, acc in results.items():
    print(f"Label {lab}: pairwise ordering accuracy = {acc:.3f}")
# (Optional) overall mean
valid_accs = [v for v in results.values() if not pd.isna(v)]
overall_acc = sum(valid_accs) / len(valid_accs)
print(f"\nOverall pairwise ordering accuracy (mean across labels): {overall_acc:.3f}")