# dnnls\_final\_project





This project is my final assessment for my Deep learning course at Sheffield Hallam University.





# Multimodal Sequence Modelling



This project focuses on multimodal sequence prediction and the improvements I have chosen to make are Text encoder and Visual encoder. The goal is for the model to predict the next step in a sequence of the visual and textual inputs used.



* **Task:** Image + Text → Next Step Prediction
* **Dataset:** StoryReasoning - Oliveira, D. A. P., \& Matos, D. M. (2025). StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation. arXiv preprint arXiv:2505.10292. https://arxiv.org/abs/2505.10292 
* **Objective:** Improve a baseline model using enhanced encoders ans optimized training parameters.



## Baseline Architecture

The baseline model consists of the following components:

* Imported Libraries
* Data preparation
* Creating the dataset and dataset loaders
* Text encoder : LSTM
* Visual encoder: CNN
* Attention Module
* Sequence predictor model
* Training, initialization and training loop



## Baseline Setup and Minor Modifications

I made some initial changes without changing the architecture :



#### I changed the drive access to local



```python

def save\_checkpoint(model, optimizer, epoch, loss, filename="autoencoder\_checkpoint.pth"):

&#x20;   """

&#x20;   Saves the checkpoint to a local folder.

&#x20;   """

&#x20;   # Local folder 

&#x20;   local\_folder = './checkpoints'



&#x20;   # Ensure the directory exists

&#x20;   os.makedirs(local\_folder, exist\_ok=True)



&#x20;   # Full file path

&#x20;   full\_path = os.path.join(local\_folder, filename)```



#### Evaluation Pipeline



