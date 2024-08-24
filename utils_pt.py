# import keras
# import keras.backend as K
# from tqdm import tqdm

# class ADAMLearningRateTracker(keras.callbacks.Callback):
#     """It prints out the last used learning rate after each epoch (useful for resuming a training)
#     original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
#     """

#     def __init__(self, end_lr):
#         super(ADAMLearningRateTracker, self).__init__()
#         self.end_lr = end_lr

#     def on_epoch_end(self, epoch, logs={}):  # works only when decay in optimizer is zero
#         optimizer = self.model.optimizer
#         # t = K.cast(optimizer.iterations, K.floatx()) + 1
#         # lr_t = K.eval(optimizer.lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /
#         #                               (1. - K.pow(optimizer.beta_1, t))))
#         # print('\n***The last Actual Learning rate in this epoch is:', lr_t,'***\n')
#         print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
#         # stops the training if the basic lr is less than or equal to end_learning_rate
#         if K.eval(optimizer.lr) <= self.end_lr:
#             print("training is finished")
#             self.model.stop_training = True


import torch
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm

class ADAMLearningRateTracker:
    def __init__(self, optimizer, end_lr):
        self.optimizer = optimizer
        self.end_lr = end_lr

    def on_epoch_end(self, epoch):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'\n***The last Basic Learning rate in this epoch is: {current_lr} ***\n')
        if current_lr <= self.end_lr:
            print("Training is finished")
            return True
        return False

# Usage example during training loop
# optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
# lr_tracker = ADAMLearningRateTracker(optimizer, end_lr=0.0001)
# for epoch in range(num_epochs):
#     train_one_epoch(...)
#     stop_training = lr_tracker.on_epoch_end(epoch)
#     if stop_training:

def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        if if_train:
            dir_type_name = "train"
            fl_img = []
            nmask = 'gt_' + filenames
            fl_msk = os.path.join(directory_name, 'train_gt', f'{nmask}.TIF')
            list_msk.append(fl_msk)
        else:
            dir_type_name = "test"
            fl_img = []
            fl_id = f'{filenames}.TIF'
            list_test_ids.append(fl_id)

        fl_img_red = os.path.join(directory_name, f'{dir_type_name}_red', f'{nred}.TIF')
        fl_img_green = os.path.join(directory_name, f'{dir_type_name}_green', f'{ngreen}.TIF')
        fl_img_blue = os.path.join(directory_name, f'{dir_type_name}_blue', f'{nblue}.TIF')
        fl_img_nir = os.path.join(directory_name, f'{dir_type_name}_nir', f'{nnir}.TIF')

        fl_img.extend([fl_img_red, fl_img_green, fl_img_blue, fl_img_nir])
        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids
