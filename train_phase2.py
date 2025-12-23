import os
import json
import random
from sklearn.utils import resample
from itertools import chain
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from model.ATP_HCR import ATP_HCR_bin_MutiQ
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects
from torchsummary import summary
import copy

"""Train ATP_HCR_bin_MutiQ model"""
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)  # CPU seed for deterministic results
    torch.cuda.manual_seed(seed)  # GPU seed for deterministic results
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)
set_seed(0)

##########################################
# Preprocessing: region extraction and mask generation
##########################################
def fine_mask(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    target_gray = 53
    gray_diff = image.astype(np.float32) - target_gray
    mask = (gray_diff < 15).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # # Visualize original mask
    # plt.imshow(mask, cmap='gray')
    # plt.title('Original Mask')
    # plt.show()

    # Find background regions (mask == 0)
    zero_mask = (mask == 0).astype(np.uint8)
    labeled_array, num_features = label(zero_mask)

    boxes = []
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        if np.any(region):
            coords = np.argwhere(region)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    if not boxes:
        # Fallback: use the full image
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        # Select the largest region
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        x, y, w, h = boxes[0]

    # Make region square and expand
    side = max(w, h) + 40
    cx = x + w // 2
    cy = y + h // 2
    new_x = max(0, cx - side // 2)
    new_y = max(0, cy - side // 2)
    max_x = min(image.shape[1], new_x + side)
    max_y = min(image.shape[0], new_y + side)
    new_w = max_x - new_x
    new_h = max_y - new_y

    return new_x, new_y, new_w, new_h

def convert_to_target_path(image_path):
    image_path = image_path.replace("\\", "/")
    # print("Original path:", image_path)
    # Define the server base path
    server_base_path = "/sibcb1/chenluonanlab8/lijinhong/zuochunman/Shared_data/Rongjianming/ShandongUniversity"
    # Get the parent of the local "image_2025" directory
    local_base_path = os.path.join("F:", "ShandongUniversity", "image_2025")
    # print("Local path:", local_base_path)
    # Replace "image_2025" with "processed_images" to build the target path
    server_base_path2 = os.path.join(server_base_path, "processed_images")
    # print("Server path:", server_base_path2)
    target_path = image_path.replace(local_base_path, server_base_path2)
    # print("Target path:", target_path)
    return target_path

def find_and_mask_smallest_region(
        image_path,
        target_color=(0x35, 0x35, 0x35),
        min_area=2500 * 2500,
        initial_tolerance=5,
        max_lighten_steps=4,
        lighten_factor=33,
        if_plot=True  # toggle visualization
):
    def get_mask_for_color(img, color, tol):
        lower = np.array([max(c - tol, 0) for c in color], dtype=np.uint8)
        upper = np.array([min(c + tol, 255) for c in color], dtype=np.uint8)
        return cv2.inRange(img, lower[::-1], upper[::-1])  # BGR order

    def enhance_mask(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def find_valid_contours(mask):
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt - 1 for cnt in contours if cv2.contourArea(cnt - 1) >= min_area]

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    original = image.copy()
    current_color = list(target_color)
    target_color_list = list(target_color)
    historical_masks = []
    lighten_step = 0

    if if_plot:  # Initial visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original image")
        plt.axis('off')

    while lighten_step <= max_lighten_steps:
        # 1. Try the current primary color
        mask = get_mask_for_color(image, current_color, initial_tolerance*(1 + lighten_step))
        enhanced_mask = enhance_mask(mask)
        contours = find_valid_contours(enhanced_mask)

        if contours:
            best_cnt = max(contours, key=lambda c: 4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))

            # Build result mask
            result_mask = np.zeros_like(enhanced_mask)
            cv2.drawContours(result_mask, [best_cnt], -1, 255, cv2.FILLED)
            result = np.full_like(image, target_color_list[::-1], dtype=np.uint8)
            result[result_mask == 255] = original[result_mask == 255]

            if if_plot:  # Visualize successful mask
                plt.subplot(1, 3, 2)
                plt.imshow(enhanced_mask, cmap='gray')
                plt.title(f"Successful mask (L={lighten_step})")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title("Final result")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            return result

        # 2. Save current mask to history
        historical_masks.append(enhanced_mask)

        # 3. Try combining all historical masks
        if lighten_step == max_lighten_steps:
            combined_mask = historical_masks[0]
            for m in historical_masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, m)

            combined_contours = find_valid_contours(combined_mask)
            if combined_contours:
                best_cnt = max(combined_contours,
                               key=lambda c: 4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))

                result_mask = np.zeros_like(combined_mask)
                cv2.drawContours(result_mask, [best_cnt], -1, 255, cv2.FILLED)
                result = np.full_like(image, current_color[::-1], dtype=np.uint8)
                result[result_mask == 255] = original[result_mask == 255]

                if if_plot:  # Visualize combined result
                    plt.subplot(1, 3, 2)
                plt.imshow(combined_mask, cmap='gray')
                plt.title(f"Combined mask (L={lighten_step})")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title("Combined result")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
                return result

        # 4. Lighten the target color and retry
        current_color = [min(c + lighten_factor, 255) for c in current_color]
        lighten_step += 1
        # print(f"Lighten step {lighten_step}: new color {current_color}")

    # All attempts failed
    print("No valid contour found after all attempts")
    if if_plot:
        plt.subplot(1, 3, 2)
        if historical_masks:
            plt.imshow(historical_masks[-1], cmap='gray')
        else:
            plt.imshow(np.zeros_like(image[:, :, 0]), cmap='gray')
        plt.title("Final attempted mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Unprocessed result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    return None

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

class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True, bin_edges=None,num_bins=8,concat_bin_num=3):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        # Raw ATP values and their log10
        self.atp_raw = [self.mapping[p] for p in self.image_paths]
        self.log_values = [np.log10(v) for v in self.atp_raw]

        self.transform = transform

        # 1) Determine bin_edges (log10-based)
        if bin_edges is None:
            lo, hi = min(self.log_values), max(self.log_values)
            bin_edges = np.linspace(lo - 1e-9, hi + 1e-9, num_bins + 1)
        self.bin_edges = bin_edges

        # 2) Compute class labels per sample (0-5 -> 6 classes)
        def assign_bin(l):
            idx = np.digitize(l, self.bin_edges)
            return 0 if idx <= concat_bin_num else (idx - concat_bin_num)
        self.labels = [assign_bin(l) for l in self.log_values]
        self.num_classes = len(set(self.labels))

        # 3) Record log10 range per class
        self.label_ranges = {}
        self.label_ranges[0] = (self.bin_edges[0], self.bin_edges[concat_bin_num])
        for i in range(concat_bin_num, len(self.bin_edges)-1):
            self.label_ranges[i-concat_bin_num+1] = (self.bin_edges[i], self.bin_edges[i+1])

        # 4) Compute percentile position within each class interval
        self.percentiles = []
        for logv, lab in zip(self.log_values, self.labels):
            low, high = self.label_ranges[lab]
            pct = (logv - low) / (high - low + 1e-12)
            pct = float(np.clip(pct, 0.0, 1.0))
            self.percentiles.append(pct)

        # Print stats before balancing
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        print("Class counts before balancing:")
        for label, count in sorted(class_counts.items()):
            low, high = self.label_ranges[label]
            print(f"Class {label}: {count} samples — ATP range: {10**low:.2f} - {10**high:.2f}")
        print("-" * 50)

        # 5) Optional class balancing
        if balance_classes:
            self._balance_classes()
            # Print stats after balancing
            class_counts = {}
            for label in self.labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            print("Class counts after balancing:")
            for label, count in sorted(class_counts.items()):
                low, high = self.label_ranges[label]
                print(f"Class {label}: {count} samples — ATP range: {10**low:.2f} - {10**high:.2f}")
            print("-" * 50)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image path
        path = convert_to_target_path(self.image_paths[idx])
        path = path.replace("image_2025", "processed_images")
        img = cv2.imread(path, 0)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        # Return image plus three labels
        cls_label = torch.tensor(self.labels[idx], dtype=torch.long)
        pct_label = torch.tensor(self.percentiles[idx], dtype=torch.float32)
        atp_raw_label = torch.tensor(self.atp_raw[idx], dtype=torch.float32)
        return img, cls_label, pct_label, atp_raw_label

    def _balance_classes(self):
        from collections import defaultdict
        by_class = defaultdict(list)
        for i, lab in enumerate(self.labels):
            by_class[lab].append(i)
        max_n = max(len(indices) for indices in by_class.values())

        new_paths, new_labels, new_pcts, new_raws = [], [], [], []
        orig_raw = self.atp_raw
        orig_pcts = self.percentiles
        orig_labels = self.labels

        for lab, indices in by_class.items():
            if len(indices) < max_n:
                sampled = random.choices(indices, k=max_n)
            else:
                sampled = indices
            for i in sampled:
                new_paths.append(self.image_paths[i])
                new_labels.append(self.labels[i])
                new_pcts.append(orig_pcts[i])
                new_raws.append(orig_raw[i])

        # Shuffle
        perm = np.random.permutation(len(new_paths))
        self.image_paths = [new_paths[i] for i in perm]
        self.labels = [new_labels[i] for i in perm]
        self.percentiles = [new_pcts[i] for i in perm]
        self.atp_raw = [new_raws[i] for i in perm]



    def _shuffle_data(self):
        """Shuffle data order"""
        indices = np.arange(len(self.balanced_image_paths))
        np.random.shuffle(indices)
        self.balanced_image_paths = [self.balanced_image_paths[i] for i in indices]
        self.balanced_labels = [self.balanced_labels[i] for i in indices]

    def _print_class_distribution(self, class_counts):
        """Print balanced class counts"""
        print("After class balancing:")
        new_class_counts = {}
        for label in self.balanced_labels:
            new_class_counts[label] = new_class_counts.get(label, 0) + 1
        for cls, count in sorted(new_class_counts.items()):
            print(f"Class {cls}: {count} samples - raw {class_counts[cls]} samples")
        print("-" * 50)

    def save_process_image(self,image,image_raw_path):
        """
        Save processed image
        :param image:
        :param image_raw_path:
        :return:
        """
        # Path for processed images, replacing "image_2025" with "processed_images"
        processed_image_path = image_raw_path.replace("image_2025", "processed_images")
        # Create directory
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        # Ensure image is numpy array (required by OpenCV)
        if isinstance(image, Image.Image):  # when given a PIL image
            image = np.array(image)
            # PIL uses RGB; OpenCV expects BGR. Save grayscale for brightfield images.
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Save in TIFF format
        cv2.imwrite(processed_image_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        # print(f"Saved processed image: {processed_image_path}")

##########################################
# Training/validation loop (same as train.py)
##########################################
def train_model(model,
                dataloaders,
                combined_criterion,
                optimizer,
                lr_schedule,
                label_ranges: dict,
                num_epochs=25,
                device='cuda',
                model_name="model"):
    """
    Train a model that performs 6-class classification and percentile regression.
    Computes:
      - pct MAE (percentile space)
      - atp MAE (original ATP space)

    Each dataloader batch returns (inputs, cls_labels, pct_labels, atp_raw_labels)
    label_ranges: {bin_idx: (low_log, high_log), ...}
    """
    model = model.to(device)
    best_mae_atp = float('inf')
    best_reg_loss = float('inf')
    best_mae_pct = float('inf')
    best_acc = 0.0

    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_loss_cls', 'val_loss_cls',
        'train_loss_reg', 'val_loss_reg',
        'train_acc', 'val_acc',
        'train_mae_pct', 'val_mae_pct',
        'train_mae_atp', 'val_mae_atp',
        'train_mae_first_atp', 'val_mae_first_atp'
    ]}
    cls_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            # if epoch == 0 and phase == 'train':
            #     continue
            model.train() if phase == 'train' else model.eval()

            running = {k: 0.0 for k in ['loss', 'loss_cls','loss_reg', 'corrects', 'abs_pct', 'abs_atp','abs_first_atp']}
            n = len(dataloaders[phase].dataset)
            per_class_mae = {i: [] for i in range(len(label_ranges))}
            # Per-class ATP percentage error (%)
            per_class_pct_err = {i: [] for i in range(len(label_ranges))}

            with tqdm(dataloaders[phase], desc=phase, unit='batch') as pbar:
                for inputs, cls_labels, pct_labels, atp_raw in pbar:
                    inputs = inputs.to(device)
                    cls_labels = cls_labels.to(device)
                    pct_labels = pct_labels.to(device)
                    atp_raw = atp_raw.to(device).float().view(-1)
                    log10atp_raw = torch.log10(atp_raw)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        atp_pred, logits,pcts,atp_max_prob_pred = model(inputs)
                        bin_probs = F.softmax(logits, dim=1)
                        atp_pred = atp_pred.view(-1)
                        atp_max_prob_pred = atp_max_prob_pred.view(-1)
                        log10atp_pred = torch.log10(atp_pred)
                        # loss = combined_criterion(log10atp_pred.squeeze(),log10atp_raw)
                        preds = bin_probs.argmax(dim=1)
                        pct_preds = pcts[torch.arange(pcts.size(0)), preds]
                        loss, loss_cls, loss_reg = combined_criterion(logits, cls_labels, pct_preds, pct_labels)
                        # loss_cls = cls_loss(bin_probs, cls_labels)
                        # loss_reg = reg_loss(pct_preds, pct_labels) * 100.00

                        # 1) MAE in percentile space
                        abs_pct = torch.abs(pct_preds.view(-1) - pct_labels).sum().item()

                        # 2) MAE in original ATP scale
                        abs_atp = torch.abs(atp_pred - atp_raw).sum().item()
                        abs_first_atp = torch.abs(atp_max_prob_pred - atp_raw).sum().item()
                        for i in range(len(atp_pred)):
                            cls = int(preds[i].item())
                            # Skip MAE accumulation if class prediction is wrong
                            if preds[i] != cls_labels[i]:
                                continue
                            err = torch.abs(atp_pred[i] - atp_raw[i]).item()
                            per_class_mae[cls].append(err)
                            low, high = label_ranges[cls]
                            range_atp = 10 ** high - 10 ** low
                            pct_err = (err / (range_atp + 1e-8)) * 100
                            per_class_pct_err[cls].append(pct_err)


                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    bs = inputs.size(0)
                    running['loss'] += loss.item() * bs
                    running['loss_cls'] += loss_cls.item() * bs
                    running['loss_reg'] += loss_reg.item() * bs
                    running['corrects'] += (preds == cls_labels).sum().item()
                    running['abs_pct'] += abs_pct
                    running['abs_atp'] += abs_atp
                    running['abs_first_atp'] += abs_first_atp

                    pbar.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'loss_cls': f"{loss_cls.item():.3f}",
                        'loss_reg': f"{loss_reg.item():.3f}",
                        'acc': f"{(preds == cls_labels).float().mean().item():.3f}",
                        'mae_pct': f"{(abs_pct / bs):.3f}",
                        'mae_atp': f"{(abs_atp / bs):.0f}",
                        'mae_first_atp': f"{(abs_first_atp / bs):.0f}"
                    })

            # epoch metrics
            epoch_loss = running['loss'] / n
            epoch_loss_cls = running['loss_cls'] / n
            epoch_loss_reg = running['loss_reg'] / n
            epoch_acc = running['corrects'] / n
            epoch_mae_pct = running['abs_pct'] / n
            epoch_mae_atp = running['abs_atp'] / n
            epoch_mae_first_atp = running['abs_first_atp'] / n

            print(f"{phase} | Loss: {epoch_loss:.4f}"
                    f" | Loss_cls: {epoch_loss_cls:.4f}"
                    f" | Loss_reg: {epoch_loss_reg:.4f}"
                  f" | Acc: {epoch_acc:.4f}"
                  f" | MAE_pct: {epoch_mae_pct:.4f}"
                  f" | MAE_atp: {epoch_mae_atp:.0f}"
                  f" | MAE_first_atp: {epoch_mae_first_atp:.0f}")

            for k, val in zip(
                    ['loss','loss_cls','loss_reg', 'acc', 'mae_pct', 'mae_atp', 'mae_first_atp'],
                    [epoch_loss,epoch_loss_cls,epoch_loss_reg, epoch_acc, epoch_mae_pct, epoch_mae_atp]
            ):
                history[f'{phase}_{k}'].append(val)

            # Print per-class ATP MAE
            class_mae_str = " | ".join(
                [f"C{cls}: {np.mean(errs):.1f}" if errs else f"C{cls}: -"
                 for cls, errs in per_class_mae.items()]
            )
            print(f"Per-class ATP MAE → {class_mae_str}")
            class_pct_str = " | ".join(
                [f"C{cls}: {np.mean(errs):.1f}%" if errs else f"C{cls}: -"
                 for cls, errs in per_class_pct_err.items()]
            )
            print(f"Per-class ATP %Error → {class_pct_str}")

            # Save best model (minimal val ATP MAE)
            if phase == 'val' and epoch_mae_atp < best_mae_atp:
                best_mae_atp = epoch_mae_atp
                best_mae_pct = epoch_mae_pct
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"./trained_models/{model_name}_best.pth")
                print(f"    → Saved best model (val MAE_atp: {best_mae_atp:.1f} | val MAE_pct: {best_mae_pct:.4f} | val ACC: {best_acc:.4f})")

        lr_schedule.step()

    print(f"\nBest val MAE_atp: {best_mae_atp:.1f}")

    # Plot metrics
    epochs = range(1, num_epochs + 1)

    def plot(name, ylabel):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history[f'train_{name}'], label='Train')
        plt.plot(epochs, history[f'val_{name}'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f"./plts/{model_name}_{name}.png")
        plt.close()

    plot('loss', 'Loss')
    plot('loss_cls', 'Loss (cls)')
    plot('loss_reg', 'Loss (reg)')
    plot('acc', 'Accuracy')
    plot('mae_pct', 'MAE (percentile)')
    plot('mae_atp', 'MAE (ATP)')

    return model


class CombinedLoss(nn.Module):
    """
    Composite loss: cross-entropy + α * (SmoothL1 or MSE) regression loss
    """
    def __init__(self, alpha=100.0, reg_loss='smoothl1',beta=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if reg_loss == 'mse':
            self.reg_criterion = nn.MSELoss()
        else:
            self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, logits, cls_labels, pct_preds, pct_labels):
        loss_cls = self.cls_criterion(logits, cls_labels)
        # If pct_preds is (B,1), squeeze first
        pct_preds = pct_preds.view(-1)
        loss_reg = self.reg_criterion(pct_preds, pct_labels)
        loss = self.beta * loss_cls + self.alpha * loss_reg
        return loss, self.beta * loss_cls, self.alpha * loss_reg

##########################################
# Main workflow
##########################################
if __name__ == '__main__':
    # Configuration
    mapping_file = './data/processed_image_atp_mapping.json'
    model_path = './trained_models/AD_logbin_best_class9.pth'
    model_path = './trained_models/AD_MutiHead_class12_best_44473515.pth'
    image_size = 512  # must match model constructor
    batch_size = 16
    num_workers = 1
    epoch_num = 50
    num_bins = 12
    model_name = "AD_MutiHead_class"+str(num_bins)  # model name
    # Class counts per checkpoint: 6→4, 7→5, 8→6, 9→7, 10→8, 11→8, 12→9, 13→10, 14→11
    print("Training model name:",model_name)

    # Load and split mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        full_mapping = json.load(f)

    # Shuffle all image paths
    image_paths = list(full_mapping.keys())
    np.random.shuffle(image_paths)

    # Split train/val
    split = int(0.75 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Create mapping dicts
    train_mapping = {path: full_mapping[path] for path in train_paths}
    val_mapping = {path: full_mapping[path] for path in val_paths}

    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30,fill=0x35),  # random 30-degree rotation
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    # Build datasets - training uses augmentation, validation uses base transform
    bin_edge,concat_bin_num = compute_bin_edges(full_mapping,num_bins=num_bins)
    print("Bin edges:", bin_edge)
    print("concat_bin_num:",concat_bin_num)
    label_ranges = {}
    label_ranges[0] = (bin_edge[0], bin_edge[concat_bin_num])
    for i in range(concat_bin_num, len(bin_edge) - 1):
        label_ranges[i - concat_bin_num + 1] = (bin_edge[i], bin_edge[i + 1])
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True, bin_edges=bin_edge,num_bins=num_bins,concat_bin_num=concat_bin_num)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False, bin_edges=bin_edge,num_bins=num_bins,concat_bin_num=concat_bin_num)

    # Determine number of classes
    num_category = train_dataset.num_classes

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = ATP_HCR_bin_MutiQ(
        in_channel=1,  # RGB input
        hidden=256,  # hidden dimension
        category=num_category,
        image_size=image_size,
        patches=64,
        label_ranges=label_ranges,
    ).to(device)
    print(model)
    # Load pretrained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
        print(f"Loaded pretrained model: {model_path}")

    # Assumes image size 224x224, 3 channels
    # summary(model, input_size=(1, image_size, image_size))

    # Loss function and optimizer (multi-class)
    criterion = CombinedLoss()

    # optimizer = torch.optim.Adam(chain(*[head.parameters() for head in model.reg_heads]), lr=1e-4, weight_decay=1e-5)
    paras = chain(model.reg_heads.parameters(), model.classifier.parameters())
    optimizer = torch.optim.Adam(paras, lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=0)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device),model_name=model_name,label_ranges=train_dataset.label_ranges)

    # Save best model
    # torch.save(model.state_dict(), 'ATP_HCR_best.pth')