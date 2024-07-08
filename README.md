# ElderNet
![ElderNet PipeLine](/imgs/ElderNetPipeline.png)    
This is the implementation of the paper: [Automated Gait Detection in Daily Living Settings from a Wrist-Worn Accelerometer in Older Adults Using Self-Supervised Learning](https://www.researchsquare.com/article/rs-4102403/v1).  
Here, we developed and evaluated a gait detection deep learning approach, termed ElderNet, that was oriented and optimized for older adults and, in particular, those who might have impaired gait. The first stage involved the training of an SSL model, utilizing the pre-trained [UK Biobank model](https://arxiv.org/abs/2206.02909) (Copyright © 2022, University of Oxford)  
This SSL model was extensively modified in both architecture and training cohorts to include a large unlabeled dataset of more than 1000 older adults with and without impaired gait who wore a wrist-worn device for up to 10 days and were participating in the [RUSH Memory and Aging Project](https://www.radc.rush.edu/home.htm;jsessionid=B4B2994C88E8AC9ED3737F576F1BE36E). Next, we fine-tuned the model on a labeled dataset consisting of 83 older adults from the [Mobilise-D](https://mobilise-d.eu/) data, each wearing a wrist-worn accelerometer for approximately 2.5 hours. 

# Load Pre-trained Models

## Overview
The ElderNet model is available with two sets of pre-trained weights:
1. **ElderNet SSL Model:** Pre-trained during the SSL stage using the MAP dataset.
2. **ElderNet Fine-tuned Model:** Further refined for gait detection using the Mobilise-D dataset.

## Example Usage
Below is an example demonstrating how to load these models using PyTorch:

```python
import torch

# Data is in windows of 10 seconds, resampled to 30Hz (i.e., 300 samples per window)
x = np.random.rand(1, 3, 300)
x = torch.FloatTensor(x)

# Load the ElderNet SSL model
ssl_model = torch.hub.load('yonbrand/ElderNet', 'eldernet_ssl', pretrained=True)
ssl_output = ssl_model(x)

# Load the ElderNet Fine-tuned model
ft_model = torch.hub.load('yonbrand/ElderNet', 'eldernet_ft')
ft_output = ft_model(x)
