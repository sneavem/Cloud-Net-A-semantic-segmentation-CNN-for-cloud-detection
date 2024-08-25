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

def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=1000):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nlwir = 'lwir_' + filenames

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
        fl_img_lwir = os.path.join(directory_name, f'{dir_type_name}_lwir', f'{nlwir}.TIF')

        fl_img.extend([fl_img_red, fl_img_green, fl_img_blue, fl_img_lwir])
        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids
