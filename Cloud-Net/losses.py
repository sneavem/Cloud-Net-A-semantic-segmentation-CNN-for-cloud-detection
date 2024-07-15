# from keras import backend as K

# smooth = 0.0000001


# def jacc_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

import torch
import torch.nn.functional as F

smooth = 0.0000001


def jacc_coef(y_true, y_pred):
    # Flatten the tensors
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    # Calculate the intersection
    intersection = torch.sum(y_true_f * y_pred_f)
    
    # Calculate the sum of the true and predicted tensors
    sum_true_pred = torch.sum(y_true_f) + torch.sum(y_pred_f)
    
    # Calculate the Jaccard coefficient
    jaccard = (intersection + smooth) / (sum_true_pred - intersection + smooth)
    
    # Return the Jaccard distance (1 - Jaccard coefficient)
    return 1 - jaccard
    
    
# def jacc_coef(y_true, y_pred):
#     """
#     Calculate the Jaccard loss between true and predicted masks.
#     """
#     # Flatten label and prediction tensors
#     true = y_true.view(-1)
#     pred = y_pred.view(-1)
    
#     # Calculate Intersection and Union
#     intersection = (true * pred).sum(dim=1)
#     total = (true + pred).sum(dim=1)
#     union = total - intersection
    
#     # Calculate the Jaccard - intersection over union
#     jaccard = (intersection + smooth) / (union + smooth)
    
#     # Return the Jaccard loss
#     return 1 - jaccard.mean()



# Usage example
# y_true and y_pred should be torch tensors
# y_true = torch.tensor([...], dtype=torch.float32)
# y_pred = torch.tensor([...], dtype=torch.float32)
# loss = jacc_coef(y_true, y_pred)
