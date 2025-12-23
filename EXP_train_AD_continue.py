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
from model.ATP_HCR import ATP_HCR_Log_Continue
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects
import copy

"""
Use the AD architecture for direct continuous-value prediction.
"""
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)  # CPU: deterministic random numbers
    torch.cuda.manual_seed(seed)  # GPU: deterministic random numbers for current GPU
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)
set_seed(0)


def convert_to_target_path(image_path):
    image_path = image_path.replace("\\", "/")
    # print("Original path:", image_path)
    # Define the server base path
    server_base_path = "/sibcb1/chenluonanlab8/lijinhong/zuochunman/Shared_data/Rongjianming/ShandongUniversity"
    # Get the parent directory of the local path (where "image_2025" resides)
    local_base_path = os.path.join("F:", "ShandongUniversity", "image_2025")
    # print("Local base path:", local_base_path)
    # Replace "image_2025" with "processed_images" to build the target path
    server_base_path2 = os.path.join(server_base_path, "processed_images")
    # print("Server path:", server_base_path2)
    target_path = image_path.replace(local_base_path, server_base_path2)
    # print("Target path:", target_path)
    return target_path

# Compute bin_edge
def compute_bin_edges(mapping_dict, num_bins=8):
    """
    Compute log10(ATP) bin edges based on the original ATP mapping.

    Args:
        mapping_dict: dict mapping image paths to ATP values
        num_bins: int, number of bins (default 8 bins, i.e., 9 edges)

    Returns:
        bin_edges: np.ndarray of length num_bins + 1
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)

    # Count samples in each bin
    bin_counts = np.histogram(log_atp_values, bins=bin_edges)[0]

    # Find the largest bin size and compute 10% of it
    max_bin_count = max(bin_counts)
    threshold = max_bin_count * 0.1

    # Accumulate from the first bins until exceeding the threshold
    cumulative = 0
    concat_bin_num = 0
    for count in bin_counts:
        cumulative += count
        concat_bin_num += 1
        if cumulative >= threshold:
            break
    return bin_edges,concat_bin_num

##########################################
# Dataset definition (mirrors ATPDataset)
##########################################
class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        self.labels = [self.mapping[p] for p in self.image_paths]
        self.balanced_classes = balance_classes
        self.transform = transform

        # Take log10 of ATP
        self.labels = [np.log10(label) for label in self.labels]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_path = convert_to_target_path(image_path)
        # # Save path for processed images, replacing "image_2025" with "processed_images"
        # image_path = image_path.replace("image_2025", "processed_images")
        try:
            # Read grayscale image
            image = cv2.imread(image_path,0)
            if image is None:
                print(f"Unable to read image: {image_path}")
                return None

            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.transform:
                # Ensure processed_img is a NumPy array
                if not isinstance(image, np.ndarray):
                    image = np.array(image)

                # Convert to PIL image
                image = Image.fromarray(image)
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


    def _shuffle_data(self):
        """Shuffle the data order."""
        indices = np.arange(len(self.balanced_image_paths))
        np.random.shuffle(indices)
        self.balanced_image_paths = [self.balanced_image_paths[i] for i in indices]
        self.balanced_labels = [self.balanced_labels[i] for i in indices]

    def _print_class_distribution(self, class_counts):
        """Print class counts after balancing."""
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
        # Save path for the processed image, replacing "image_2025" with "processed_images"
        processed_image_path = image_raw_path.replace("image_2025", "processed_images")
        # Create directory
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        # Ensure image is a NumPy array (OpenCV needs this)
        if isinstance(image, Image.Image):  # If it is a PIL image
            image = np.array(image)
            # PIL images are usually RGB, while OpenCV expects BGR; save bright-field images as grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Save as TIFF
        cv2.imwrite(processed_image_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        # print(f"Saved processed image: {processed_image_path}")

##########################################
# Training/validation loop (same structure as train.py)
##########################################
def train_model(model, dataloaders, criterion, optimizer,lr_schedule, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_loss = float('inf')

    # Initialize history containers
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
            # if epoch == 0 and phase == 'train':
            #     continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Clear predictions and labels each epoch
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
                        preds = outputs.squeeze()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    running_loss += loss.item() * inputs.size(0)

                    # Collect predictions and labels
                    epoch_labels.extend(labels.detach().cpu().numpy())
                    epoch_preds.extend(preds.detach().cpu().numpy())

                    pbar.set_postfix({
                        'loss': f"{loss.item():.2f}",
                    })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './trained_models/AD_continue_best_class.pth')
                print(f"Best model saved with loss: {best_loss:.4f}")
        lr_schedule.step()

    # Plot train/validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plts/AD_logbin_loss_plot_class.png')
    plt.show()

    return model


##########################################
# Main workflow
##########################################
if __name__ == '__main__':
    # Configure parameters
    mapping_file = './data/processed_image_atp_mapping.json'
    model_path = './AD_logbin_best_0.8925.pth'
    image_size = 512  # Must match the model constructor
    batch_size = 16
    num_workers = 1
    epoch_num = 60

    # Load and split mapping_file
    with open(mapping_file, 'r', encoding='utf-8') as f:
        full_mapping = json.load(f)

    # Gather all image paths and shuffle
    image_paths = list(full_mapping.keys())
    np.random.shuffle(image_paths)

    # Split train/validation sets
    split = int(0.75 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # Build mapping dictionaries
    train_mapping = {path: full_mapping[path] for path in train_paths}
    val_mapping = {path: full_mapping[path] for path in val_paths}

    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30,fill=0x35),  # Random 30-degree rotation
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

    # Create datasets - train uses augmented transforms, val uses base transforms
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False)


    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = ATP_HCR_Log_Continue(
        in_channel=1,  # RGB input
        hidden=256,  # Hidden dimension
        image_size=image_size,
        patches=64
    ).to(device)
    # Load pretrained model
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"Loading pretrained model: {model_path}")
    # Assume image size is 224x224, 3 channels
    # summary(model, input_size=(1, image_size, image_size))

    # Loss function and optimizer (multiclass)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device))

    # Save best model
    # torch.save(model.state_dict(), 'ad2d_mil_best.pth')