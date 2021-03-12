# 3D-CNN-LUNA16
This is a test code to train 3D CNN model on LUNA16 database
The original code is from this repository https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py
I just modified it to load my .mhd data from my files.
It's a binary classification problem, I edited the truth table as in the following code https://github.com/Mustafa-MS/compare-csv
I got an accuracy ~ 75% for train and test.
I think the problem is that the model is affected by the noise from other big organs besides the lungs.
Next step is to train a scanning window CNN, first by cropping the nodules, train the model, then build a scanning window.
