import os

## folder names
ASSIGNMENT_DATA_ROOT = "data/assignment_data_bdd"
IMAGES_DIR = os.path.join(
    ASSIGNMENT_DATA_ROOT, "bdd100k_images_100k/bdd100k/images/100k"
)
LABELS_DIR = os.path.join(ASSIGNMENT_DATA_ROOT, "bdd100k_labels_release/bdd100k/labels")
NEW_LABELS_DIR = os.path.join(
    ASSIGNMENT_DATA_ROOT, "bdd100k_images_100k/bdd100k/labels/100k"
)

## label files
TRAIN_LABEL_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_train.json")
VAL_LABEL_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_val.json")
NEW_TRAIN_LABEL_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_train.jsonl")
NEW_VAL_LABEL_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_val.jsonl")
