import os
import json
import random
from sklearn.utils import resample

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from model.ATP_HCR import ATP_HCR
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects
import copy


plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# Set the random seed
def set_seed(seed):
    torch.manual_seed(seed)  # CPU seed for deterministic random numbers
    torch.cuda.manual_seed(seed)  # GPU seed for deterministic random numbers
    torch.backends.cudnn.deterministic = True  # cuDNN
    np.random.seed(seed)  # NumPy
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

    # # Visualize the raw mask
    # plt.imshow(mask, cmap='gray')
    # plt.title('Original Mask')
    # plt.show()

    # Locate background regions (mask == 0)
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
        # Fallback: use the whole image
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        # Pick the largest region
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = boxes[0]

    # Convert to a square and expand
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
    # Get the local base path (where "image_2025" resides)
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
        if_plot=True  # Additional flag to control visualization
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

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image: {image_path}")
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

            # Create the result
            result_mask = np.zeros_like(enhanced_mask)
            cv2.drawContours(result_mask, [best_cnt], -1, 255, cv2.FILLED)
            result = np.full_like(image, target_color_list[::-1], dtype=np.uint8)
            result[result_mask == 255] = original[result_mask == 255]

            if if_plot:  # Visualize success
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

        # 2. Save the current mask to history
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

                if if_plot:  # Visualize the combined result
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

        # 4. Lighten the target color
        current_color = [min(c + lighten_factor, 255) for c in current_color]
        lighten_step += 1
        # print(f"Lighten step {lighten_step}: new color {current_color}")

    # All attempts failed
    print("No valid contour found in any attempt")
    if if_plot:
        plt.subplot(1, 3, 2)
        if historical_masks:
            plt.imshow(historical_masks[-1], cmap='gray')
        else:
            plt.imshow(np.zeros_like(image[:, :, 0]), cmap='gray')
        plt.title("Final attempt mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Unprocessed result")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    return None

# Compute bin_edge
def compute_bin_edges(mapping_dict, num_bins=8):
    """
    Compute bin boundaries on log10(ATP) based on the original mapping.

    Args:
        mapping_dict: dict mapping image paths to ATP values
        num_bins: int, number of bins (default 8 bins = 9 edges)

    Returns:
        bin_edges: np.ndarray of length num_bins + 1
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)
    return bin_edges

##########################################
# Dataset definition (similar to ATPDataset)
##########################################
class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True, bin_edges=None):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        self.labels = [self.mapping[p] for p in self.image_paths]
        self.balanced_classes = balance_classes
        self.transform = transform

        # Take log10 of ATP
        self.labels = [np.log10(label) for label in self.labels]

        # Standardize bin_edges
        if bin_edges is None:
            min_label, max_label = min(self.labels), max(self.labels)
            bin_edges = np.linspace(min_label - 1e-9, max_label + 1e-9, 9)
        self.bin_edges = bin_edges

        # Print sample counts per bin
        print("Number of log10(ATP) labels in each bin:")
        for i in range(len(self.bin_edges) - 1):
            count = sum(1 for l in self.labels if self.bin_edges[i] <= l < self.bin_edges[i + 1])
            print(f"Bin {i}: {count} labels")

        # Assign labels (merge first 3 bins into class 0, others increment)
        def assign_bin(l):
            bin_idx = np.digitize(l, self.bin_edges)  # bin_idx ∈ [1, len]
            if bin_idx <= 3:
                return 0
            return bin_idx - 3  # Class indices start from 1

        self.labels = [assign_bin(l) for l in self.labels]
        self.num_classes = len(set(self.labels))

        # Build label_ranges: merge first 3 bins into class 0, others keep separate indices
        self.label_ranges = {}
        self.label_ranges[0] = (self.bin_edges[0], self.bin_edges[3])
        for i in range(3, len(self.bin_edges) - 1):
            self.label_ranges[i - 2] = (self.bin_edges[i], self.bin_edges[i + 1])

        print("ATP range per class (log10(ATP)):")
        for label in sorted(self.label_ranges):
            low, high = self.label_ranges[label]
            print(f"Class {label}: log10(ATP) ∈ ({low:.4f}, {high:.4f})")

        # Optional class balancing
        if balance_classes:
            self._balance_classes()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_path = convert_to_target_path(image_path)
        # # Processed image save path, replace "image_2025" with "processed_images"
        # image_path = image_path.replace("image_2025", "processed_images")
        try:
            # Read the image in grayscale
            image = cv2.imread(image_path, 0)
            if image is None:
                print(f"Unable to read image: {image_path}")
                return None

            label = torch.tensor(self.labels[idx], dtype=torch.long)

            if self.transform:
                # Ensure processed_img is a numpy array
                if not isinstance(image, np.ndarray):
                    image = np.array(image)

                # Convert to a PIL image
                image = Image.fromarray(image)
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

    def _balance_classes(self):
        """Sample each class up to max_count to balance data"""
        # Use the original data as the base
        original_image_paths = self.image_paths
        original_labels = self.labels

        # Count original samples per class
        from collections import defaultdict
        class_to_samples = defaultdict(list)
        for img_path, label in zip(original_image_paths, original_labels):
            class_to_samples[label].append(img_path)

        # Determine the maximum class count
        max_count = max(len(samples) for samples in class_to_samples.values())

        # Build balanced data lists
        self.balanced_image_paths = []
        self.balanced_labels = []

        for label, samples in class_to_samples.items():
            num_samples = len(samples)
            if num_samples < max_count:
                # Sample with replacement for this class
                resampled = random.choices(samples, k=max_count)
            else:
                resampled = samples  # If already at max count, do not duplicate

            self.balanced_image_paths.extend(resampled)
            self.balanced_labels.extend([label] * max_count)

        # Replace the original data
        self.image_paths = self.balanced_image_paths
        self.labels = self.balanced_labels

        # Shuffle the data order
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        # Output the new sample count per class
        print("After class balancing:")
        for label in sorted(class_to_samples.keys()):
            print(f"Class {label}: {max_count} samples - raw {len(class_to_samples[label])} samples")
        print("-" * 50)

    def _shuffle_data(self):
        """Shuffle the data order"""
        indices = np.arange(len(self.balanced_image_paths))
        np.random.shuffle(indices)
        self.balanced_image_paths = [self.balanced_image_paths[i] for i in indices]
        self.balanced_labels = [self.balanced_labels[i] for i in indices]

    def _print_class_distribution(self, class_counts):
        """Print class counts after balancing"""
        print("After class balancing:")
        new_class_counts = {}
        for label in self.balanced_labels:
            new_class_counts[label] = new_class_counts.get(label, 0) + 1
        for cls, count in sorted(new_class_counts.items()):
            print(f"Class {cls}: {count} samples - raw {class_counts[cls]} samples")
        print("-" * 50)

    def save_process_image(self,image,image_raw_path):
        """
        Save the processed image
        :param image:
        :param image_raw_path:
        :return:
        """
        # Save path for processed images; replace "image_2025" with "processed_images"
        processed_image_path = image_raw_path.replace("image_2025", "processed_images")
        # Create directories as needed
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        # Ensure image is a numpy array (required by OpenCV)
        if isinstance(image, Image.Image):  # If it's a PIL image
            image = np.array(image)
            # PIL images are usually RGB; convert to grayscale for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Save as TIFF
        cv2.imwrite(processed_image_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        # print(f"Saved processed image: {processed_image_path}")

##########################################
# Training and validation flow (aligned with train.py)
##########################################
def train_model(model, dataloaders, criterion, optimizer,lr_schedule, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_acc = 0.4
    best_model_wts = None

    # Initialize history trackers
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{num_epochs}")


        for phase in ['train', 'val']:
            if epoch == 0 and phase == 'train':
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Reset predictions and labels for each epoch
            epoch_labels = []
            epoch_preds = []

            with tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch + 1}', unit='batch') as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Collect predictions and labels
                    epoch_labels.extend(labels.cpu().numpy())
                    epoch_preds.extend(preds.cpu().numpy())

                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}"
                    })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())

            if phase == 'val':
                # Save validation predictions for confusion matrix
                all_labels = epoch_labels
                all_preds = epoch_preds

                # Print classification report
                from sklearn.metrics import classification_report
                print("\nClassification Report:")
                print(classification_report(all_labels, all_preds))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'AD_logbin_best.pth')
                print(f"Best model saved with accuracy: {best_acc:.4f}")
        lr_schedule.step()

    print(f"Best val Acc: {best_acc:.4f}")
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('AD_logbin_loss_plot.png')
    plt.show()
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('AD_logbin_accuracy_plot.png')
    plt.show()

    return model


##########################################
# Main routine
##########################################
if __name__ == '__main__':
    # Configuration
    mapping_file = 'processed_image_atp_mapping.json'
    model_path = './AD_logbin_best_0.8925.pth'
    image_size = 512  # Must match the model constructor
    batch_size = 16
    num_workers = 1
    epoch_num = 30
    num_bins = 8

    # image_size = 1024  # Must match the model constructor
    # batch_size = 4
    # num_workers = 1
    # epoch_num = 30

    # Load and split the mapping file
    with open(mapping_file, 'r', encoding='utf-8') as f:
        full_mapping = json.load(f)

    # Shuffle all image paths
    image_paths = list(full_mapping.keys())
    np.random.shuffle(image_paths)

    # Split training and validation sets
    split = int(0.75 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Build mapping dictionaries for each split
    train_mapping = {path: full_mapping[path] for path in train_paths}
    val_mapping = {path: full_mapping[path] for path in val_paths}

    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30, fill=0x35),  # Random 30° rotation
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

    # Build datasets: training uses augmentation, validation uses base transform
    bin_edge = compute_bin_edges(full_mapping,num_bins=num_bins)
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True, bin_edges=bin_edge)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False, bin_edges=bin_edge)

    # Number of categories
    num_category = train_dataset.num_classes

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = ATP_HCR(
        in_channel=1,  # RGB input
        hidden=256,  # Hidden dimension
        category=num_category,
        num_layer=4,  # Number of feature extraction layers
        image_size=image_size,
        patches=64
    ).to(device)
    # Load a pretrained model
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"Loaded pretrained model: {model_path}")
    # Assume image size 224x224 with 3 channels
    # summary(model, input_size=(1, image_size, image_size))

    # Loss function and optimizer (multiclass)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device))

    # Save the best model
    # torch.save(model.state_dict(), 'ad2d_mil_best.pth')