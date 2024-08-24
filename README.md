# Cloud-Net: A semantic segmentation CNN for cloud detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cloud-net-an-end-to-end-cloud-detection-1/semantic-segmentation-on-38-cloud)](https://paperswithcode.com/sota/semantic-segmentation-on-38-cloud?p=cloud-net-an-end-to-end-cloud-detection-1)

## Training Cloud-Net on [38-Cloud Training Set](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)

### Requirements
Run `python3 

### Scripts
Run ```python main_train.py``` to train the network on 38-Cloud training set. The path to the dataset folder should be set at ```GLOBAL_PATH = 'path to 38-cloud dataset'```. The directory tree for the dataset looks like as following:

├──38-Cloud dataset

│------------├──Cloud-Net_trained_on_38-Cloud_training_patches.h5

│------------├──Training

│------------------├──Train blue<br/>
                      .
                      .
                      .

│------------------├──training_patches_38-cloud.csv

│------------├──Test

│------------------├──Test blue<br/>
                      .
                      .
                      .

│------------------├──test_patches_38-cloud.csv

│------------├──Predictions


The training patches are resized to 192 * 192 before each iteration. Then, four corresponding spectral bands are stacked together to create a 192 * 192 * 4 array. A ```.log``` file is generated to keep track of the loss values. The loss function used for training is the soft Jaccard loss.

## Testing Cloud-Net on [38-Cloud Test Set](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
Run ```python main_test.py``` for getting the predictions. The weights of Cloud-Net, pretrained on 38-Cloud training set, is available [here: Cloud-Net_trained_on_38-Cloud_training_patches.h5](https://vault.sfu.ca/index.php/s/2Xk6ZRbwfnjrOtu). Relocate this file in the dataset directory as shown above. The predicted cloud masks will be generated in the "Predictions" folder. Then, use the [Evaluation over 38-Cloud Dataset section](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset#evaluation-over-38-cloud-dataset) to get the numerical results and precited cloud masks for the entire scenes. 
