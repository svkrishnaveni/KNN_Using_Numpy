#!/usr/bin/env python

import numpy as np
import math
import re

#Loading program data separated as features and targets
def Load_2c2d3a3d_traindata(str_path_2c2d3c3d_program, remove_age = False):
    '''
    inputs: str path to train data.txt
    outputs: numpy arrays of features, targets
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(str_path_2c2d3c3d_program) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 3 features 
            lsFeature_tmp = [float(data_tmp.split(',')[0]),float(data_tmp.split(',')[1]),int(data_tmp.split(',')[2])]
            # extract target as string
            lsTarget_tmp = [data_tmp.split(',')[3][1]]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
        if remove_age:            
            for i in features:
                del i[2]
    targets = [x[0] for x in targets]
    return np.array(features),np.array(targets)

#Loading test data with features
def Load_2a3a_testdata(str_path_2a3a_test):
    '''
    inputs: str path to test data.txt
    outputs: test features 
    '''
    # initialize empty lists to gather features and targets
    features = []
    # read lines in txt file as string
    with open(str_path_2a3a_test) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 3 features 
            lsFeature_tmp = [float(data_tmp.split(',')[0]),float(data_tmp.split(',')[1]),int(data_tmp.split(',')[2])]
            features.append(lsFeature_tmp)
    return np.array(features)

#Loading train data separated as features and targets
def Load_2a3a_traindata(str_path_2a3a_train):
    '''
    inputs: str path to train data.txt
    outputs: numpy arrays of features, targets
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(str_path_2a3a_train) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 3 features 
            lsFeature_tmp = [float(data_tmp.split(',')[0]),float(data_tmp.split(',')[1]),int(data_tmp.split(',')[2])]
            # extract target as string
            lsTarget_tmp = [data_tmp.split(',')[3][1]]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
    targets = [x[0] for x in targets]
    return np.array(features),np.array(targets)


#Gaussian probability distribution for x
def gaussian_probability_density(x,mean,var):
    '''
    This function calculates gaussian probability density
        inputs: Distance from the mean(x), Mean of the distribution, Variance of the distribution
        outputs: Probability density at x
    '''
    exponent=np.exp(-((x-mean)**2)/(2*var))
    probability=(1/(math.sqrt(2*np.pi*var)))*exponent
    return probability


#calculating mean
def mean(list_numbers):
    '''
    This function calculates mean of given list of numbers
        inputs: List of numbers
        outputs: Mean of given list of numbers
    '''
    return sum(list_numbers)/len(list_numbers)


#calculating variance
def variance(list_numbers):
    '''
    This function calculates variance of given list of numbers
          inputs: List of numbers
          outputs: Varinace of given list of numbers
    '''
    avg=mean(list_numbers)
    variance=sum((i-avg)**2 for i in list_numbers)/(len(list_numbers)-1)
    return variance

def gaussian_naive_bayes(features,labels,sample):
    '''
    This function performs classification using Gaussian Naive Bayes algorithm
    inputs: array(train features), array(labels), 1 test sample row
    outputs: predicted class label
    '''
    # initialize P(class/Data) list
    p_cls_given_data = []
    for cls in list(np.unique(labels)):
        class_ind = np.where(labels == cls)
        p_cls = len(class_ind[0])/len(labels)
        # 'mean' function gives a vector of feature means
        # 'variance' function gives a vector of all feature variances
        m = mean(features[class_ind])
        v = variance(features[class_ind])
        # compute gaussian probability densities for each feature in sample for each class
        GPD =[gaussian_probability_density(sample[column],m[column],v[column]) for column in list(range(features.shape[1]))]
        # compute probability of class given data
        p_cls_given_data.append(p_cls*(np.prod(GPD)))

    return np.unique(labels)[np.argmax(p_cls_given_data)]



def cartesian_distance(a,b):
    '''
    This function calculates euclidean distance between 2 vectors
    inputs:Two vectors
    outputs:Euclidean Distance between given vectors
    '''
    distance=np.sqrt(np.sum(np.square(a-b)))
    return distance

#Function for KNN classifier which returns distance to k neighbors and their tagets for given test sample
def knn_classifier(train_data,train_labels,sample,k):
    '''
    This function is KNN classifier
    inputs:
        train_features = mxn array (m= #observations,n= #features)
        train_labels = nx1 array of targets
        sample = 1 row of test features
        k =  int (#nearest neighbors)
    outputs:
        Euclidean distance of k neighbors
        Targets of k nearest neighbors
    '''
    distance=[]
    neighbors=[]
    for i in range(len(train_data)):
        d=cartesian_distance(train_data[i],sample)
        distance.append(d)
    ind = np.argsort(distance)
    distance.sort()
    for i in range(k):
        neighbors.append(distance[i])
    targets = [train_labels[i] for i in ind[:k]]   
    return neighbors,targets

#Performance evaluation using leave one out cross validation for KNN
def leave_one_out_knn(train_data,train_labels,k):
    '''
    This function validates KNN algorithm with leave one out scheme
    input:
        train_data = mxn array (m= #observations,n= #features)
        train_labels = nx1 array of targets
        k =int value
    output:percentage of true predicted values for given k value
    '''

    preds =[]
    for i in range(train_data.shape[0]):
        sample = train_data[i]
        # removing row of test sample from training data
        train_data_tmp = np.delete(train_data,i,0)
        train_labels_tmp = np.delete(train_labels,i)
        n,t = knn_classifier(train_data_tmp,train_labels_tmp,sample,k)
        elt=max(set(t), key=t.count)
        preds.append(elt)
    counter=0
    for i in range(len(preds)):
        if(preds[i]==train_labels[i]):
            counter=counter+1
    accuracy=(counter/len(preds))*100
    return accuracy

#Performance evaluation using leave one out cross validation for Gaussian Naive Bayes
def leave_one_out_gnb(train_data,train_labels):
    '''
    This function validates GNB algorithm with leave one out scheme
    inputs:
         inputs: array(train features), array(labels)
    output:percentage of true predicted values
    '''
    preds =[]
    for i in range(train_data.shape[0]):
        sample = train_data[i]
        # removing row of test sample from training data
        train_data_tmp = np.delete(train_data,i,0)
        train_labels_tmp = np.delete(train_labels,i)
        t = gaussian_naive_bayes(train_data_tmp,train_labels_tmp,sample)
        elt=max(set(t), key=t.count)
        preds.append(elt)
    counter=0
    for i in range(len(preds)):
        if(preds[i]==train_labels[i]):
            counter=counter+1
    accuracy=(counter/len(preds))*100
    return accuracy