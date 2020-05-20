# rad_chestxray
This is a repo for anyone wants to apply AI/DL in diagnosing diseases from Chest-Xray Images

# Directory structure
To modulize the repo, the current structure from @hoangvu is adopted as follows:
```bash 
├── checkpoint # Where to save the model 
│   └── 
├── dataflow # Customize dataloader 
│   ├── dataset.py # Load and Pre-processing data
│   └── transforms.py # Data Augmentation
├── losses # Customize losses 
│   └── BCELogLoss.py
├── metrics # Customize Metrics 
│   └── __init__.py
├── models # Customize model 
│   ├── __init__.py
├── noisy_labels # Apply "Confident Learning" to filter outliers from Multi-(class/label) images classification task 
│   ├── correct_noisy_labels.py
├── optim # Customize Optimizers
│   └── __init__.py
├── README.md
└── train.py # main file will be store here 
```


# Customize Dataset with native Pytorch Dataloader
PyTorch provides many tools to make data loading easy and hopefully, to make our code more readable. In this scenario, we will use Dataset class to load data from a non trivial dataset (labeled Vinmec).

### Dataset structure
```bash 
├── data # folder which contains all images for train/validation 
│   └── 00015cc7-2591-4380-bbe6-6abf0d5e9768.dcm.bmp
│   └── 00086612-768f-446a-8d81-ddfad434fbdf.dcm.bmp
├── train.csv #  which contains training set with Images's column and diseases's columns (19)
├── valid.csv #  which contains valid set with Images's column and diseases's columns (19)
├── test.csv #  which contains test set with Images's column and diseases's columns (19)
```
For detail of csv files, please review it and get insights before training models.

### Dataloader
```bash
├── dataflow # Customize dataloader 
│   ├── dataset.py
```

# Customize model 
```python
from models import Classifier
```

# Customize Optimization 
```python
from optim import create_optimizer
```

# Customize loss 
```python
from losses import init_loss_func
```

# Command to train 
##Train Chest X-Ray Classification
```python 
python train.py --save_path=/path/to/save/ckpt 
                --save_top_k=8 //save top checkpoints on the given metric
                --gpus=0 # specified gpu device
                --data_path=/path/to/dataset
                --log-every=50
                --test-every=1000 
                --json_path=cfg.json 
                --epochs=8
```

These above command is packaged by using the following bash file:
```bash 
./train.sh
```

# Details of configuration in cfg.json 
Advanced option can be controlled in configuration file cfg.json
Note that the hyperparameters hparams will be dumped to this cfg.json as well
```json 
{
  "backbone": "densenet201", //CNN architecture
  "gray": false, # Run model with gray images
  "use_se": false,
  "img_size": 256,
  "crop_size": 224,
  "pixel_mean": 128.0, //normalized mean
  "pixel_std": 64.0, //normalized standard deviation
  "crop": false, //randomly ResizedCrop image for pre-processing
  "n_crops": 0, // 0 (no crop) - 5 (five-crops) - 10 (ten-crops)
  "imagenet": true, // normalize images
  "extract_fields": "0,1,2,3,4", //Specific fields needed to extract (default: 5 main diseases)
  "offset": 1, //index of first disease column in csv file
  "upsample_index": [], //use Upsampling technique for data augmentation
  "upsample_times": 1,
  "train_batch_size": 32, //training batch size
  "dev_batch_size": 32, //valid batch size
  "pretrained": true, //use or not use pretrained backbone models
  "norm_type": "BatchNorm", //Normalization type for CNNs
  "global_pool": "AVG_MAX", //Global pooling type such as AVG (Average), MAX (max), ... (default=AVG_MAX)
  "fc_bn": true, //use or not use batch normalization after Global Pooling
  "attention_map": "FPA", //Global Attention Map such as FPA, SAM, ...
  "lse_gamma": 0.5, //paramter for Global LogSumExpPool
  "fc_drop": 0, //dropout before classifier
  "optim": "adam", //Optimizer for training
  "criterion": "bce", //Loss type for training
  "lr_scheduler": "step", //Step, Cosin, ...
  "lr": 0.0001, //Initialized learning rate
  "lr_factor": 0.1, //Reduce learning rate with specific factor
  "step_size": 2, //reduce learning after specific number of epochs
  "momentum": 0.9, //momentum factor for SGD
  "weight_decay": 0.0, //Initialized weight decay for optimizers
  "beta1": 0.9, //coefficient used for computing running averages of gradient and its square
  "beta2": 0.999, //coefficient used for computing running averages of gradient and its square
  "threshold": 0.5, //Fix threshold for calculating True Positives, ...
  "metric_type": "weighted", //Average metric type such as micro, macro, weighted which is used for Precision, Recall, Fbeta_Score
  "beta": 2 //Weight of precision in harmonic mean which is used for fbeta_score
}
```


# Private evaluation
The private evaluation will use click interface and some input parameters as below:
```python 
@click.command()
@click.option("--prediction", default='predictions.csv',
                help="Predictions from model which only contains binary values")
@click.option("--labels", default='labels.csv', 
                help="Ground truth labels")
@click.option("--out", default='.', 
                help="Output path")
@click.option("--beta", default=1, 
                help="parameter for f_beta_score")
```
Command line to run
```bash 
python private_eval.py --prediction=test_preds.csv --labels=valid.csv --beta=1
```

# Model Deployment
```python
TODO
```