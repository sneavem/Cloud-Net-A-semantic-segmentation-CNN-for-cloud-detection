
import torch
import torch.nn.functional as F

smooth = 0.0000001


def jacc_coef(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = torch.sum(y_true_f * y_pred_f)
    sum_true_pred = torch.sum(y_true_f) + torch.sum(y_pred_f)
    jaccard = (intersection + smooth) / (sum_true_pred - intersection + smooth)
    
    # Return the Jaccard distance (1 - Jaccard coefficient)
    return 1 - jaccard
    
    
# Usage example
# y_true and y_pred should be torch tensors
# y_true = torch.tensor([...], dtype=torch.float32)
# y_pred = torch.tensor([...], dtype=torch.float32)
# loss = jacc_coef(y_true, y_pred)
