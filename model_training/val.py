from ultralytics import YOLO
from tqdm import tqdm
from dataloader import BDDValDataset, yolo_collate_fn, yolo_uncollate_fn, get_boxes
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
import json
import pickle
from torchvision.ops import box_iou
import cv2
import numpy as np
import os


def convert_tensor_to_python(obj):
    """
    Convert PyTorch tensors to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain tensors
        
    Returns:
        Converted object with tensors as Python types
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "item") and obj.numel() == 1:
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_tensor_to_python(v) for k, v in obj.items()}
    else:
        return obj


def rescale_box(box, scale_x, scale_y):
    """
    Rescale bounding box coordinates.
    
    Args:
        box: Bounding box coordinates
        scale_x (float): X-axis scaling factor
        scale_y (float): Y-axis scaling factor
        
    Returns:
        list: Rescaled box coordinates
    """
    return [
        box[0].item() * scale_x,
        box[1].item() * scale_y,
        box[2].item() * scale_x,
        box[3].item() * scale_y,
    ]


def process_predictions(pred_result, device):
    """
    Process model predictions into standardized format.
    
    Args:
        pred_result: Raw prediction results from model
        device: PyTorch device (cuda/cpu)
        
    Returns:
        dict: Processed predictions with boxes, scores, labels
    """
    if len(pred_result) > 0:
        boxes = pred_result[:, :4]
        conf = pred_result[:, 4]
        classes = pred_result[:, 5].long()
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        conf = torch.zeros((0,), dtype=torch.float32)
        classes = torch.zeros((0,), dtype=torch.long)

    return {
        "boxes": boxes.to(device),
        "scores": conf.to(device),
        "labels": classes.to(device),
    }


def process_targets(targets, batch_idx, device):
    """
    Process ground truth targets for a specific batch index.
    
    Args:
        targets: Ground truth targets tensor
        batch_idx (int): Batch index to process
        device: PyTorch device (cuda/cpu)
        
    Returns:
        dict: Processed targets with boxes, scores, labels
    """
    mask = targets[:, 0] == batch_idx
    target_boxes = targets[mask]

    if len(target_boxes) > 0:
        boxes = target_boxes[:, 2:6]
        conf = torch.ones(len(target_boxes), dtype=torch.float32)
        classes = target_boxes[:, 1].long()
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        conf = torch.zeros((0,), dtype=torch.float32)
        classes = torch.zeros((0,), dtype=torch.long)

    return {
        "boxes": boxes.to(device),
        "scores": conf.to(device),
        "labels": classes.to(device),
    }


def find_matches(pred_boxes, pred_labels, tgt_boxes, tgt_labels, iou_threshold=0.5):
    """
    Find matched predictions and targets based on IoU and class agreement.
    
    Args:
        pred_boxes: Predicted bounding boxes
        pred_labels: Predicted class labels
        tgt_boxes: Target bounding boxes
        tgt_labels: Target class labels
        iou_threshold (float): IoU threshold for matching
        
    Returns:
        tuple: Sets of matched prediction and target indices
    """
    matched_preds = set()
    matched_targets = set()

    if len(pred_boxes) > 0 and len(tgt_boxes) > 0:
        ious = box_iou(pred_boxes, tgt_boxes)
        for p_idx in range(len(pred_boxes)):
            for t_idx in range(len(tgt_boxes)):
                if (
                    ious[p_idx, t_idx] >= iou_threshold
                    and pred_labels[p_idx] == tgt_labels[t_idx]
                ):
                    matched_preds.add(p_idx)
                    matched_targets.add(t_idx)
                    break

    return matched_preds, matched_targets


def collect_correct_predictions(
    pred_boxes, pred_labels, pred_scores, matched_preds, scale_x, scale_y
):
    """
    Collect correctly predicted bounding boxes.
    
    Args:
        pred_boxes: Predicted bounding boxes
        pred_labels: Predicted class labels
        pred_scores: Prediction confidence scores
        matched_preds: Set of matched prediction indices
        scale_x (float): X-axis scaling factor
        scale_y (float): Y-axis scaling factor
        
    Returns:
        list: List of correct prediction dictionaries
    """
    correct_predictions = []
    for p_idx in range(len(pred_boxes)):
        if p_idx in matched_preds:
            rescaled_box = rescale_box(pred_boxes[p_idx], scale_x, scale_y)
            correct_predictions.append(
                {
                    "box": rescaled_box,
                    "label": int(pred_labels[p_idx]),
                    "confidence": float(pred_scores[p_idx]),
                }
            )
    return correct_predictions


def collect_missed_boxes(tgt_boxes, tgt_labels, matched_targets, scale_x, scale_y):
    """
    Collect missed target bounding boxes.
    
    Args:
        tgt_boxes: Target bounding boxes
        tgt_labels: Target class labels
        matched_targets: Set of matched target indices
        scale_x (float): X-axis scaling factor
        scale_y (float): Y-axis scaling factor
        
    Returns:
        list: List of missed box dictionaries
    """
    missed_boxes = []
    for t_idx in range(len(tgt_boxes)):
        if t_idx not in matched_targets:
            rescaled_box = rescale_box(tgt_boxes[t_idx], scale_x, scale_y)
            missed_boxes.append({"box": rescaled_box, "label": int(tgt_labels[t_idx])})
    return missed_boxes


def collect_additional_boxes(
    pred_boxes, pred_labels, pred_scores, matched_preds, scale_x, scale_y
):
    """
    Collect additional (false positive) predicted bounding boxes.
    
    Args:
        pred_boxes: Predicted bounding boxes
        pred_labels: Predicted class labels
        pred_scores: Prediction confidence scores
        matched_preds: Set of matched prediction indices
        scale_x (float): X-axis scaling factor
        scale_y (float): Y-axis scaling factor
        
    Returns:
        list: List of additional prediction dictionaries
    """
    additional_boxes = []
    for p_idx in range(len(pred_boxes)):
        if p_idx not in matched_preds:
            rescaled_box = rescale_box(pred_boxes[p_idx], scale_x, scale_y)
            additional_boxes.append(
                {
                    "box": rescaled_box,
                    "label": int(pred_labels[p_idx]),
                    "confidence": float(pred_scores[p_idx]),
                }
            )
    return additional_boxes


def plot_predictions_on_image(
    img_path,
    correct_preds,
    missed_boxes,
    tgt_boxes,
    tgt_labels,
    output_dir="data/validation_data_visualization",
):
    """
    Plot predictions and ground truth on an image and save to output directory.
    
    Args:
        img_path (str): Path to the image file
        correct_preds (list): List of correct predictions
        missed_boxes (list): List of missed detections
        tgt_boxes: Ground truth bounding boxes
        tgt_labels: Ground truth labels
        output_dir (str): Directory to save visualized images
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    scale_x = 1280 / 640
    scale_y = 720 / 640

    for box in correct_preds:
        x1, y1, x2, y2 = [int(coord) for coord in box["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Correct: {box['label']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    for box in missed_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"Missed: {box['label']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    for i, (box, label) in enumerate(zip(tgt_boxes, tgt_labels)):
        x1, y1, x2, y2 = [
            int(coord * scale)
            for coord, scale in zip(box, [scale_x, scale_y, scale_x, scale_y])
        ]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            img,
            f"GT: {int(label)}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img)


def format_metrics_text(map_results, iou_results):
    """
    Format evaluation metrics into readable text format.
    
    Args:
        map_results: Mean Average Precision results
        iou_results: Intersection over Union results
        
    Returns:
        str: Formatted metrics text
    """
    text_output = []
    text_output.append("=" * 60)
    text_output.append("VALIDATION METRICS RESULTS")
    text_output.append("=" * 60)
    text_output.append("")

    text_output.append("Mean Average Precision (mAP):")
    text_output.append("-" * 30)
    map_dict = convert_tensor_to_python(map_results)
    for key, value in map_dict.items():
        if isinstance(value, (int, float)):
            text_output.append(f"  {key}: {value:.4f}")
        else:
            text_output.append(f"  {key}: {value}")
    text_output.append("")

    text_output.append("Intersection over Union (IoU):")
    text_output.append("-" * 30)
    iou_dict = convert_tensor_to_python(iou_results)
    for key, value in iou_dict.items():
        if isinstance(value, (int, float)):
            text_output.append(f"  {key}: {value:.4f}")
        else:
            text_output.append(f"  {key}: {value}")
    text_output.append("")
    text_output.append("=" * 60)

    return "\n".join(text_output)


def save_results(results1, results2, evaluation_data):
    """
    Save evaluation results to files.
    
    Args:
        results1: Mean Average Precision results
        results2: IoU results
        evaluation_data (list): List of evaluation data dictionaries
    """
    metrics_text = format_metrics_text(results1, results2)

    print(metrics_text)

    with open("data/validation_metrics.txt", "w") as f:
        f.write(metrics_text)

    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/evaluation_data.json", "w") as f:
        json.dump(evaluation_data, f, indent=2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    plot_images = True
    scale_x = 1280 / 640
    scale_y = 720 / 640
    iou_threshold = 0.01

    model = YOLO("data/models/bdd_detector.pt")
    val_dataset = BDDValDataset()
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=yolo_collate_fn
    )

    metric1 = MeanAveragePrecision(iou_type="bbox").to(device)
    metric2 = IntersectionOverUnion().to(device)

    evaluation_data = []

    for imgs, targets, names in tqdm(
        val_loader, desc="Evaluation on validation dataset..."
    ):
        with torch.no_grad():
            results = model.predict(imgs, verbose=False, conf=0.01)
        pred_batch_result = yolo_uncollate_fn(results)

        preds = []
        for pred_result in pred_batch_result:
            preds.append(process_predictions(pred_result, device))

        tgts = []
        batch_size = len(set(targets[:, 0].int().tolist()))
        for batch_idx in range(batch_size):
            tgts.append(process_targets(targets, batch_idx, device))

        for i, (pred, tgt, name) in enumerate(zip(preds, tgts, names)):
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()

            tgt_boxes = tgt["boxes"].cpu()
            tgt_labels = tgt["labels"].cpu()

            matched_preds, matched_targets = find_matches(
                pred_boxes, pred_labels, tgt_boxes, tgt_labels, iou_threshold
            )


            correct_predictions = collect_correct_predictions(
                pred_boxes, pred_labels, pred_scores, matched_preds, scale_x, scale_y
            )
            missed_boxes = collect_missed_boxes(
                tgt_boxes, tgt_labels, matched_targets, scale_x, scale_y
            )
            additional_boxes = collect_additional_boxes(
                pred_boxes, pred_labels, pred_scores, matched_preds, scale_x, scale_y
            )

            evaluation_data.append(
                {
                    "image_name": name,
                    "correct_predictions": correct_predictions,
                    "missed_boxes": missed_boxes,
                    "additional_boxes": additional_boxes,
                    "total_predictions": len(pred_boxes),
                    "total_targets": len(tgt_boxes),
                }
            )

            if plot_images:
                img_path = os.path.join(val_dataset.img_dir, name)
                plot_predictions_on_image(
                    img_path,
                    correct_predictions,
                    missed_boxes,
                    tgt_boxes.cpu(),
                    tgt_labels.cpu(),
                )

        metric1.update(preds, tgts)
        metric2.update(preds, tgts)

    results1 = metric1.compute()
    results2 = metric2.compute()

    save_results(results1, results2, evaluation_data)
