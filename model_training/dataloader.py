import os
import json
from torch.utils.data import Dataset
from config import *
from PIL import Image
import torchvision.transforms as T
import torch

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_class_map():
    """
    Load class mapping from JSON file.
    
    Returns:
        dict: Class name to index mapping
    """
    with open("data/metadata/category_map.json", "r") as f:
        class_mapping = json.load(f)
    return class_mapping


def get_boxes(labels, class_map, w, h, target_size=640):
    """
    Convert labels to YOLO format bounding boxes.
    
    Args:
        labels (list): List of label dictionaries
        class_map (dict): Class name to index mapping
        w (int): Original image width
        h (int): Original image height
        target_size (int): Target image size for scaling
        
    Returns:
        torch.Tensor: Tensor of bounding boxes in YOLO format
    """
    b = []
    scale_x = target_size / w
    scale_y = target_size / h
    for obj in labels:
        if "box2d" not in obj:
            continue
        x1, y1, x2, y2 = obj["box2d"].values()
        cls = class_map[obj["category"]]
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        b.append([-1, cls, x1_scaled, y1_scaled, x2_scaled, y2_scaled])
    if not b:
        return torch.zeros((0, 6), dtype=torch.float32)
    return torch.tensor(b, dtype=torch.float32)


def create_train_transforms():
    """
    Create training image transformations.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )


def create_val_transforms():
    """
    Create validation image transformations.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )


def create_test_transforms():
    """
    Create test image transformations.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )


def yolo_collate_fn(batch):
    """
    Collate function for YOLO training batches.
    
    Args:
        batch (list): List of batch items
        
    Returns:
        tuple: Batched images, targets, and names
    """
    images, targets, names = zip(*batch)
    images = torch.stack(images, 0)
    for i, target in enumerate(targets):
        target[:, 0] = i
    targets = torch.cat(list(targets), 0)
    return images, targets, names


def yolo_uncollate_fn(results):
    """
    Uncollate YOLO prediction results.
    
    Args:
        results: YOLO prediction results
        
    Returns:
        list: List of bbox results
    """
    bbox_results = []
    for idx, result in enumerate(results):
        bbox_results.append(result.boxes.data)
    return bbox_results


def parse_input(json_string):
    """
    Parse JSON string input to extract image name and labels.
    
    Args:
        json_string (str): JSON string containing image data
        
    Returns:
        tuple: Image name and labels list
    """
    obj = json.loads(json_string)
    name = obj["name"]
    labels = obj["labels"]
    return name, labels


class BDDTrainDataset(Dataset):
    """
    PyTorch Dataset class for BDD100k training data.
    """
    
    def __init__(self):
        """
        Initialize the training dataset.
        """
        super().__init__()
        self.img_dir = os.path.join(IMAGES_DIR, "train")
        self.label_dir = LABELS_DIR
        self.img_list = os.listdir(self.img_dir)
        with open(os.path.join(NEW_TRAIN_LABEL_FILE), "r") as f:
            self.label_list = f.readlines()
        self.transforms = create_train_transforms()
        self.class_map = get_class_map()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        name, labels = parse_input(self.label_list[idx])

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        original_width, original_height = 1280, 720
        img = self.transforms(img)

        boxes = get_boxes(labels, self.class_map, original_width, original_height)
        return img, boxes, name


class BDDValDataset(Dataset):
    """
    PyTorch Dataset class for BDD100k validation data.
    """
    
    def __init__(self):
        """
        Initialize the validation dataset.
        """
        super().__init__()
        self.img_dir = os.path.join(IMAGES_DIR, "val")
        self.label_dir = LABELS_DIR
        self.img_list = os.listdir(self.img_dir)
        with open(os.path.join(NEW_VAL_LABEL_FILE), "r") as f:
            self.label_list = f.readlines()
        self.transforms = create_val_transforms()
        self.class_map = get_class_map()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        name, labels = parse_input(self.label_list[idx])
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        original_width, original_height = 1280, 720
        img = self.transforms(img)
        boxes = get_boxes(labels, self.class_map, original_width, original_height)
        return img, boxes, name


class BDDTestDataset(Dataset):
    """
    PyTorch Dataset class for BDD100k test data.
    """
    
    def __init__(self):
        """
        Initialize the test dataset.
        """
        super().__init__()
        self.img_dir = os.path.join(IMAGES_DIR, "test")
        self.img_list = os.listdir(self.img_dir)
        self.transforms = create_test_transforms()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_list[idx])).convert("RGB")
        img = self.transforms(img)
        return img


