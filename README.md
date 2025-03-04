# CRNN

Trained with Synth dataset.

## Results

### CTC Loss on Validation and Test Sets

| Set        | CTC Loss |
|------------|----------|
| Validation | 0.0902   |
| Test       | 0.XXX    |


### Benchmark Performance

| Benchmark   | Lexicon-free (%) | Lexicon-based (%) |
|-------------|------------------|-------------------|
| ICDAR 2003 | XX.X             | YY.Y              |
| ICDAR 2013 | XX.X             | YY.Y              |
| ICDAR 2015 | XX.X             | YY.Y              |
|    SVT     | XX.X             | YY.Y              |
|   IIIT5K   | XX.X             | YY.Y              |

## Configs

Example config file:
```
{
    "device": "cuda",
    "batch_size": 32,
    "rho": 0.9, # we used Adadelta optimizer with rho value of 0.9 as shown in the paper
    "epoch_num": 2,
    "log_per_epoch": 6, # determin number of log to show per one epoch
    "patience_limit": 5 # if validation loss does not improve for num of patience limit stop the training
}

### Parameter Details
- rho: We used the Adadelta optimizer with a rho value of 0.9, as shown in the paper.
- log_per_epoch: Determines the number of logs to display per epoch.
- patience_limit: If the validation loss does not improve for the specified number of logging steps, training is stopped.
```

## Usage

Follow these steps to clone the repository, install dependencies, and run the training script:

1. Clone the repository
```
git clone https://github.com/your_username/CRNN.git
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run the Training Script
```
import os
import sys
sys.path.append("path_to_cloned_repo")
from scripts.train import Trainer
trainer = Trainer(path_to_config_file, checkpoints_path=checkpoints_path)
trainer.train()
```

## Citation

This project implements the CRNN model from:  
Shi, B., Bai, X., & Yao, C. (2017). "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298-2304. [DOI: 10.1109/TPAMI.2016.2646371](https://doi.org/10.1109/TPAMI.2016.2646371)