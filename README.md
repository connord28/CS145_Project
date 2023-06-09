# Group Datadogs: This repo is for CS 145's Group Project.
## Members: Harshil Bhullar, Connor Daly, Ryan Daly, Jerry Yang

For our project, we decided to develop our own CNN and gain a deeper understanding of different CNN archiectures. We attempted to figure out what allows a CNN to perform better/worse. This project was inspired from the two layer NN in HW 3.

We wanted to compare our CNN's performance against pretrained learning models, and attempt to get as close as we can to the pretrained models in terms of performance.

This notebook contains:
* DataProcessing.py: used to transform the Caltech-101 dataset to be compatible with our models.
* Train.py: our main training loop
* constants.py: contains hyper-parameters
* main.py: initialiizes dataset, model, and runs training loop
* networks: folder that contains all the different CNNs we worked on, as well as the pretrained models we utilized
* best_models: saved torch objects of our best models as a result of validation accuracy from our training loop

In addition, we used the notebook linked below as a way to utilize Google Colab's resources. The notebook basically mirrors main.py, with minor adjustments.

Link to our collab notebook: https://colab.research.google.com/drive/1JuF7BWP2WBKiqipqF0_11W0TjqvelXIW?usp=sharing
