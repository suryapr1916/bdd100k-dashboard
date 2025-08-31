import json
import numpy as np
from tqdm import tqdm


def get_missed_predictions_annotations(predictions, ground_truth):
    """
    Extract annotations for missed predictions by matching closest ground truth boxes.
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truth (list): List of ground truth annotation dictionaries
        
    Returns:
        list: List of missed prediction annotations
    """
    missed_predictions = []

    for pred_img_idx, pred_img_obj in tqdm(enumerate(predictions)):
        gt_img_obj = ground_truth[pred_img_idx].copy()
        missed_boxes = predictions[pred_img_idx]["missed_boxes"]

        gt_boxes = []
        for label in gt_img_obj["labels"]:
            if "box2d" in label:
                box = label["box2d"]
                gt_boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])

        missed_box_idxs = set()

        for missed_box in missed_boxes:
            missed_coords = np.array(missed_box["box"])
            min_distance = float("inf")
            closest_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                gt_coords = np.array(gt_box)
                distance = np.abs(missed_coords - gt_coords).sum()

                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx

            if closest_idx != -1:
                missed_box_idxs.add(closest_idx)

        gt_img_obj["labels"] = [
            label
            for i, label in enumerate(gt_img_obj["labels"])
            if i in missed_box_idxs and "box2d" in label
        ]
        missed_predictions.append(gt_img_obj)

    return missed_predictions


def get_correct_predictions_annotations(predictions, ground_truth):
    """
    Extract annotations for correct predictions by matching closest ground truth boxes.
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truth (list): List of ground truth annotation dictionaries
        
    Returns:
        list: List of correct prediction annotations
    """
    correct_predictions = []

    for pred_img_idx, pred_img_obj in tqdm(enumerate(predictions)):
        gt_img_obj = ground_truth[pred_img_idx].copy()
        correct_boxes = predictions[pred_img_idx]["correct_predictions"]

        gt_boxes = []
        for label in gt_img_obj["labels"]:
            if "box2d" in label:
                box = label["box2d"]
                gt_boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])

        correct_box_idxs = set()

        for correct_box in correct_boxes:
            correct_coords = np.array(correct_box["box"])
            min_distance = float("inf")
            closest_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                gt_coords = np.array(gt_box)
                distance = np.abs(correct_coords - gt_coords).sum()

                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx

            if closest_idx != -1:
                correct_box_idxs.add(closest_idx)

        gt_img_obj["labels"] = [
            label
            for i, label in enumerate(gt_img_obj["labels"])
            if i in correct_box_idxs and "box2d" in label
        ]
        correct_predictions.append(gt_img_obj)

    return correct_predictions


if __name__ == "__main__":

    annotations_file = "data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    predictions_file = "data/metadata/evaluation_data.json"

    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    with open(annotations_file, "r") as f:
        ground_truth = json.load(f)

    missed = get_missed_predictions_annotations(predictions, ground_truth)
    correct = get_correct_predictions_annotations(predictions, ground_truth)

    with open("data/metadata/missed.jsonl", "w") as f:
        for item in missed:
            f.write(json.dumps(item) + "\n")
    with open("data/metadata/correct.jsonl", "w") as f:
        for item in correct:
            f.write(json.dumps(item) + "\n")
