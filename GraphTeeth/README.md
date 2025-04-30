#    GraphTeeth
![alt text](GraphTeeth_pipline.png)

## Requirements

* Python 3.8
* PyTorch
* Check the required python packages in [requirements.txt](requirements.txt).
```
pip install -r requirements.txt
```

## Training and Testing

To train and test the GraphTeeth model, you can modify the required parameters in the [config/omni_config.yaml](config/omni_config.yaml) configuration file or pass them directly via the command line:

Train:
```
python train.py --exp-name 'edge_resnet34' --arc 'resnet34' --gpu_ids '0'
```

Test:

Modify the 'modelpath' variable and set it to the path of the trained model.
```
python test.py --exp-name 'edge_resnet34' --arc 'resnet34' --gpu_ids '0'
```
