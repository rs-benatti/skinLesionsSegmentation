# IMA201-SkinLesions-FerreiraSenna



## Introduction
For now we will just use the README as a TODO list for this project

### 0. Finish border remotion

### 1. Hair removal

### 2. Create google colab folder
This step is meant to use a jupyter notebook on goolge colab so we can improve the exposition of our results and code.

### 3. Create methods to compare to label masks
We need to compare our results, so, it's important to have a estatisical method that compares our masks.

### 4. Check for discontinuities and multiples instances od the masks
This step is meant tocheck if we have holes inside the mask, and eliminate them (maybe with a dilation).
We also need to find if we have multiple masks in a image. If we do we could (integrate them to the other mask?) choose the most centralized one.

### 5. Create a method to adjust hyperparameters
We need to adjust the hyperparameters to their best. A computationaly expensive but pratical method to do so is iterate through hyperparameters and check which combination gives us a better pontuation for all images based on the method created in 2.
