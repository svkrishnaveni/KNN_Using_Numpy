Requirements and dependencies:
    1. Numpy
    2. re  (regular expression)
    3. Math

Loading train and test data:

I have copied the data from web generated php files into text files (data_train_2a3a.txt, data_test_2a3a.txt, data_2c2d3c3d_program.txt)
I use custom functions (Load_2a3a_traindata, Load_2a3a_testdata, Load_2c2d3a3d_traindata) to load and convert .txt files to numpy arrays
Please place input .txt files (data_train_2a3a.txt, data_test_2a3a.txt, data_2c2d3c3d_program.txt)  in the current working directory

Running instructions:

run the following command to perform Leave one out CV using KNN for 3 features and 2 features
'python knn_evaluate.py'

run the following command to perform Leave one out CV using Gaussian Naive Bayes for 3 features and 2 features
'python gaussian_naive_bayes_evaluate.py'

Inorder to use custom data, copy the data to a text file and use Load_train* or Load_test* functions with appropriate paths to .txt files.

All supporting functions are mainly located in utilities.py
