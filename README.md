# Authentiear_2D
2D spectrogram based approach to authenticate users using sensor data from headphones.


There are two seperate notebooks. One uses a siamese network that utilizes L2 distance, we call this the basic version. The other uses triplet loss to train the siamese network. They both have the same preprocessing and image generation functions. They differ in model training and train-test set generation functions.

Some sample data is provided in the ./P01 folder.

**Basic version:**

Notebook: Athentiear_project_version.ipynb

The notebook is seperated into sections. The "in use code" section holds all the functions being used and the required import statements. The behaviour of the functions are explained using comments and sample code. The section "wrapper functions" calls all functions and combines their output. The "old code and resources" section holds versions of functions that are no in use and some links to useful resources.

To run the notebook, modify parameters in the "wrapper functions" section and run cells one by one as per the comments.

A script version of the notebook is available as model_training.py

**Triplet loss version**

Notebook: ./triplet_model/triplet_model.ipynb

To run this notebook, run cell by cell after modifying parameters as per the comments. A model architecture file and a trained model weight file is also available in the same folder.
