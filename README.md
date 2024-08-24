# U-Net for Cloud Detection in Multispectral Satellite Imagery for BeaverCube2 

## Training U-Net on [38-Cloud Training Set](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
The U-Net architecture is used to train a cloud detection model on the 38-Cloud dataset. The 38-Cloud dataset is a cloud
segmentation dataset that contains 38 Landsat 8 scenes  with corresponding cloud masks. For an 
extension to this dataset please see [95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset).

## Beavercube2 Cameras
The BeaverCube2 satellite has two cameras: a Boson LWIR camera and a Bluefox EO camera. The 
sensor resolution of the Boson camera is 320x256 pixels, and the sensor resolution of the Bluefox
camera is 752x480 pixels. The GSD of the images from the Boson camera is ~200m and the GSD of the 
images from the Bluefox camera is ~100m.

Because the Landsat 8 imagery is at a 30m resolution and LWIR bands are at 100m spatial resolution,
the training data was augmented to induce scale and rotation invariance in the network. 

### Requirements
* Download the 38-Cloudset using the link above or from this [38-Cloud Kaggle Dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images).  Then move the 38 Cloud dataset to the base directory of this repository. 
* Clone the official Vitis AI repository from [here](https://github.com/Xilinx/Vitis-AI).
 
 ## Training the U-Net Model
Next, install the required packages by running `pip3 install -r requirements.txt` on the machine 
you are using to train. In the `U-Net` directory, run `python3 lwir_download.py train` to download the LWIR 
 band for training images. Also run `python3 lwir_download.py test` to download the LWIR 
Landsat 8 band for testing. Then run `python3 main_train.py` to train the U-Net model on the 38-Cloud dataset.
The path to the dataset folder should be set at `GLOBAL_PATH = 'path to 38-cloud dataset'`. The directory tree for the dataset looks like as following:

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


The training patches are resized to 192 * 192 before each iteration. Then, four corresponding spectral 
bands are stacked together to create a 192 * 192 * 4 array. A ```.log``` file is generated to keep track of the loss values. The loss function used for training is the soft Jaccard loss. 

## Testing U-Net on [38-Cloud Test Set](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
Run `python main_test.py` for getting the predictions. The predicted cloud masks will be generated in the "Predictions" folder. 
Then, use the [Evaluation over 38-Cloud Dataset section](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset#evaluation-over-38-cloud-dataset) to get the numerical results and predicted cloud masks for the entire scenes. 

## Conversion to DPU 
See the document titled "Vitis AI Tool Use" in the Beavercube2 documentation in the Flight software 
folder for further information. 
To clone the Vitis AI repository and pull the relevant docker containers, run 
```bash
git clone https://github.com/Xilinx/Vitis-AI
docker pull xilinx/vitis-ai:latest
docker pull xilinx/vitis-ai-cpu:1.4.1.978
```
Then start the latest docker container with 
```bash
./docker_run.sh {Container name}
conda activate vitis-ai-pytorch
```
Next, use the contents of the `xilinx_comp` directory in this repo to run the Vitis AI quantizer. 
You may need to reinstall the required pip packages in the docker container. 






