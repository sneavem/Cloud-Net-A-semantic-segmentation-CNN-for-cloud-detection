import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import ADAMLearningRateTracker, get_input_image_names
from losses import jacc_coef
from generators import mybatch_generator_train, mybatch_generator_validation
import pandas as pd
from unet_model import UNet


def train():
    model = UNet(n_channels=4, n_classes=1)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=starting_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience, min_lr=end_learning_rate, verbose=True)


    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                              test_size=val_ratio,
                                                                              random_state=42, shuffle=True)
    gen_train = mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit)
    gen_valid = mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit)
    steps_per_epoch = int(np.ceil(len(train_img_split) / batch_sz))
    validation_steps = int(np.ceil(len(val_img_split) / batch_sz))

    best_loss = float('inf')

    print(f'Steps per epoch : {steps_per_epoch}')
    
    for epoch in range(max_num_epochs):
        model.train()
        train_loss = 0
        i = 0
        # for i in tqdm(range(steps_per_epoch), desc=f"{i}/{steps_per_epoch}"):
        for i in range(10): #steps_per_epoch):
            print(f'On iteration {i}')
            images, masks = next(gen_train)
            images = torch.tensor(images, dtype=torch.float32).to(device)
            masks = torch.tensor(masks, dtype=torch.float32).to(device)
            images = images.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]
            masks = masks.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]

            optimizer.zero_grad()
            outputs = model(images)
        
            loss = jacc_coef(masks, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= steps_per_epoch

        # print("eval")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # for _ in range(validation_steps):
            for _ in range(10):
                images_val, masks_val = next(gen_valid)
                images_val = torch.tensor(images_val, dtype=torch.float32).to(device)
                masks_val = torch.tensor(masks_val, dtype=torch.float32).to(device)
                images_val = images_val.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]
                masks_val = masks_val.permute(0, 3, 1, 2)  # permute to [12, 4, 192, 192]

                outputs_val = model(images_val)
                loss_val = jacc_coef(masks_val, outputs_val)
                val_loss += loss_val.item()

        val_loss /= validation_steps
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{max_num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")

        # Early stopping
        if optimizer.param_groups[0]['lr'] <= end_learning_rate:
            print("Training finished due to learning rate threshold")
            break

# Global variables
GLOBAL_PATH = '/nobackup/users/samue100/'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')

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
batch_sz = 12  # was 12
max_bit = 65535  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net"
weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.pt')
train_resume = False

# Getting input images names
train_patches_csv_name = 'training_patches_38-Cloud.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Train the model
train()
