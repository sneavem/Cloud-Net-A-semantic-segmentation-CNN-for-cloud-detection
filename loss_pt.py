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
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth))

# Usage example
# y_true and y_pred should be torch tensors
# y_true = torch.tensor([...], dtype=torch.float32)
# y_pred = torch.tensor([...], dtype=torch.float32)
# loss = jacc_coef(y_true, y_pred)
