# Faster R-CNN, Mask R-CNN, EfficientDet, DETR, Deformable DETR
This work mainly uses the [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v3.0.0) framework to train and test Faster R-CNN, Mask R-CNN, EfficientDet, DETR, and Deformable DETR.

In MMDetection, we provide some of our files with the following structure:
```
mmdetection
    ├── checkpoints                Store pretrained model weight files.
    ├── work_dirs3                 Stored the weight files of each of our trained models.([Download]())
    ├── data                       Folder for storing datasets.
    ├── mmdet
        ├── datasets               Dataset processing files.
        ├── evaluation
            ├── metrics            Metrics for dataset.
    ├── my_configs                 Our configuration files.       
    ├── omni_train_together.sh     Shell scripts for training.
    ├── omni_test_bestepoch.sh     Shell scripts for testing.                
```
The pretrained models in the 'checkpoints' folder are publicly available([Download](https://drive.google.com/file/d/19-5m9eHTPa-0NDEUF79EFBXatmxacvfT/view?usp=drive_link)). Our pretrained models are sourced from MMDetection as well as other open-source projects.

The trained model files in the 'work_dirs3' folder are available for download([Download](https://drive.google.com/file/d/1LsXoj-E94LsOZHAZ4Z5iOnfKZ4WXoamR/view?usp=drive_link)).

## Training and Testing
For detailed installation, configuration, as well as training and testing steps of mmdetection, please refer to the [official mmdetection documentation](https://github.com/open-mmlab/mmdetection/blob/v3.0.0/README.md).


