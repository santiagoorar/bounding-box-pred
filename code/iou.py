import torch
import torch.nn as nn



def iou_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    The function resturns the average IoU metric between the two bounding boxes.
    :return: a tensor of shape [b,] of the IoU metric values per sample in the batch
    """
    
    x_left = torch.max(pred[:, 0], target[:, 0]) # calculate the intersections
    y_top = torch.max(pred[:, 2], target[:, 2])
    x_right = torch.min(pred[:, 1], target[:, 1])
    y_bottom = torch.min(pred[:, 3], target[:, 3])

    # the area of the intersection
    intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

    # the area of union rectangle
    pred_area = (pred[:, 1] - pred[:, 0]) * (pred[:, 3] - pred[:, 2])
    target_area = (target[:, 1] - target[:, 0]) * (target[:, 3] - target[:, 2])
    union_area = pred_area + target_area - intersection_area

    iou = intersection_area / union_area

    # average IoU metric across the batch
    iou_avg = torch.mean(iou, dim=0)  # just to be able to use this as a loss function

    return iou_avg