import os
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from generators_pt import mybatch_generator_train, mybatch_generator_validation
from losses import jacc_coef
import pandas as pd
from utils_pt import get_input_image_names, ADAMLearningRateTracker
from pytorch_nndct.apis import torch_quantizer, dump_xmodel# , evaluate
import torch
from cloud_net_model import ModelArch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from unet_model import UNet
import argparse

in_rows = 752
in_cols = 752
num_of_channels = 4
num_of_classes = 1
starting_learning_rate = 1e-4
end_learning_rate = 1e-8
max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 1
max_bit = 65535  # maximum gray level in landsat 8 images


# GLOBAL_PATH = '/home/sammy/shit/Vitis-AI/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/Cloud-Net/38_cloud_set'
GLOBAL_PATH = '/workspace/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/Cloud-Net/38_cloud_set'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
 
# getting input images names
train_patches_csv_name = 'training_patches_38-Cloud.csv'
train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                               test_size=val_ratio,
                                                                               random_state=42, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# gen_train = mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit)
gen_valid = mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit)


def eval_model(model, validation_steps=10):
    model.eval()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=starting_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience, min_lr=end_learning_rate, verbose=True)


    val_loss = 0
    with torch.no_grad():
        for i in range(validation_steps):
            print(f'On step {i}')
            images_val, masks_val = next(gen_valid)
            images_val = torch.tensor(images_val, dtype=torch.float32).to(device)
            masks_val = torch.tensor(masks_val, dtype=torch.float32).to(device)
            images_val = images_val.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]
            masks_val = masks_val.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]

            outputs_val = model(images_val)
            print(outputs_val)
            loss_val = jacc_coef(masks_val, outputs_val)
            val_loss += loss_val.item()

    val_loss /= validation_steps
    scheduler.step(val_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize and test a UNet model.")
    parser.add_argument('mode', choices=['calib', 'test'], help="Mode of operation: 'calib' or 'test'.")
    args = parser.parse_args()

    model = UNet(n_channels=4, n_classes=1)
    filename = "Cloud-Net.pt"
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    inputs = torch.randn([1, 4, in_cols, in_rows])

    quantizer = torch_quantizer(args.mode, model, (inputs))
    quant_model = quantizer.quant_model

    eval_model(quant_model)

    # quantizer.export_torch_script()
    if args.mode == 'calib':
        quantizer.export_quant_config()
    else: 
        dump_xmodel(deploy_check=False)

