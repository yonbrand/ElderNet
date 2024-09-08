# ElderNet: Automated Gait Detection for Older Adults

![ElderNet Pipeline](/imgs/ElderNetPipeline.png)

## Overview

ElderNet is an implementation of the research paper: [Self-supervised learning of wrist-worn daily living accelerometer data improves the automated detection of gait in older adults](https://www.nature.com/articles/s41598-024-71491-3).

This project presents a deep learning approach for gait detection, specifically designed and optimized for older adults, including those with impaired gait. ElderNet utilizes self-supervised learning (SSL) and fine-tuning techniques to achieve accurate gait detection from wrist-worn accelerometer data.

## Key Features

- Self-supervised learning model based on the pre-trained UK Biobank model
- Fine-tuned for gait detection using labeled data from older adults
- Optimized for wrist-worn accelerometer data
- Suitable for daily living settings

## Model Architecture

ElderNet's development involved two main stages:

1. **SSL Model Training**: We modified the [UK Biobank model](https://arxiv.org/abs/2206.02909) (Copyright © 2022, University of Oxford) and trained it on a large unlabeled dataset from the [RUSH Memory and Aging Project](https://www.radc.rush.edu/home.htm). This dataset included more than 1000 older adults with and without impaired gait, wearing wrist-worn devices for up to 10 days.

2. **Fine-tuning**: The SSL model was then fine-tuned on a labeled dataset from the [Mobilise-D](https://mobilise-d.eu/) project, consisting of 83 older adults wearing wrist-worn accelerometers for approximately 2.5 hours each.

## Installation

### Using pip

To install the required packages using pip, run:
```
pip install -r requirements.txt
```
### Using Conda

To create a Conda environment with the required packages, run:
```
conda env create -f environment.yml
conda activate eldernet
```
Note: Some packages may require manual installation or updates depending on your system configuration.

## Usage

ElderNet provides two pre-trained models:

1. ElderNet SSL Model: Pre-trained during the SSL stage using the MAP dataset.
2. ElderNet Fine-tuned Model: Further refined for gait detection using the Mobilise-D dataset.

## Example Usage

Below is an example demonstrating how to load and use the ElderNet models:

```python
import numpy as np
import torch

def load_model(repo_name, model_name, device):
    model = torch.hub.load(repo_name, model_name, pretrained=True)
    return model.to(device)

def main():
    repo_name = 'yonbrand/ElderNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate sample data (10 seconds window, 30Hz sampling rate)
    x = torch.FloatTensor(np.random.rand(1, 3, 300)).to(device)
    
    # Load models
    ssl_model = load_model(repo_name, 'eldernet_ssl', device)
    ft_model = load_model(repo_name, 'eldernet_ft', device)
    
    # Generate outputs
    with torch.no_grad():
        ssl_output = ssl_model(x)
        ft_output = ft_model(x)
    
    print(f"SSL Model Output Shape: {ssl_output.shape}")
    print(f"Fine-tuned Model Output Shape: {ft_output.shape}")

if __name__ == "__main__":
    main()
ft_output = ft_model(x)
