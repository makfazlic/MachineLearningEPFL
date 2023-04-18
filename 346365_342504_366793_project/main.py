import argparse

# Extra imports are here for plotting and data management, never used in implementation
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, split_train_test

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    # 80     20     80       20
    xval, yval = None, None
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    data_size = len(xtrain) + len(xtest)

    #Plotting the data
    # tsvd = TruncatedSVD(n_components=10)
    # xtrain_tsvd = tsvd.fit_transform(xtrain)
    # tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate=100)
    # xtrain_tsne = tsne.fit_transform(xtrain_tsvd)
    # df = pd.DataFrame(xtrain_tsne, columns=['x', 'y'])
    # df['label'] = ytrain
    # sns.scatterplot(x='x', y='y', hue='label', data=df)
    # plt.show()

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        xtrain, xtest, ytrain, ytest = split_train_test(xtrain, ytrain, test_size=0.2)

    print(f"[INFO] Data loaded: xtrain.shape = {xtrain.shape} - ytrain.shape = {ytrain.shape}")
    print(f"[INFO] Data loaded: xtest.shape = {xtest.shape} - ytest.shape = {ytest.shape}")
    print(f"[INFO] Data composition: train = {len(xtrain)/data_size:.2f} - test = {len(xtest)/data_size:.2f}")

    ### WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)

    elif args.method == "svm":

        argC = args.svm_c
        argGamma = args.svm_gamma
        argKernel = args.svm_kernel
        argDegree = args.svm_degree
        argCoef0 = args.svm_coef0

        method_obj = SVM(argC, argKernel, argGamma, argDegree, argCoef0)

    ## Report results: performance on train and valid/test sets
    # acc = accuracy_fn(preds_train, ytrain)
    # macrof1 = macrof1_fn(preds_train, ytrain)
    # print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    # acc = accuracy_fn(preds, ytest)
    # macrof1 = macrof1_fn(preds, ytest)
    # print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    if not args.test:
        l_params = method_obj.validate_linear(xtrain, ytrain, xtest, ytest)
        print("[LINEAR] Best hyperparameters found: C = {}, gamma = {}".format(l_params[0], l_params[1]))
        method_obj = SVM(l_params[0], "linear", l_params[1])
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"[LINEAR] Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        r_params = method_obj.validate_rbf(xtrain, ytrain, xtest, ytest)
        print("[RBF] Best hyperparameters found: C = {}, gamma = {}".format(r_params[0], r_params[1]))
        method_obj = SVM(r_params[0], "rbf", r_params[1])
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"[RBF] Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        p_params = method_obj.validate_poly(xtrain, ytrain, xtest, ytest)
        print("[POLY] Best hyperparameters found: C = {}, gamma = {}, degree = {}, coef0 = {}".format(p_params[0], p_params[1], p_params[2], p_params[3]))
        method_obj = SVM(p_params[0], "poly", p_params[1], p_params[2], p_params[3])
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"[POLY] Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        print(l_params)
        print(r_params)
        print(p_params)

    else:
        preds_train = method_obj.fit(xtrain, ytrain)
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Train set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")



if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    # Feel free to add more arguments here if you need!

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
