# Fatma Usta - December 2020
import os
import re
import time
import argparse
import logging
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def read_data():
    """

    Reads training and test files and returns them as dataframe.

    Parameters
    ----------
    None

    Returns
    -------
    train_df : pandas dataframe
        Training dataset
    test_df : pandas dataframe
        Test dataset
    """

    # set up argument parser
    parser = argparse.ArgumentParser(description='Machine Learning Project Arguments.')
    parser.add_argument('--train_file',
                        dest='train_file',
                        action='store',
                        required=True,
                        help='path to the training dataset.')
    parser.add_argument('--test_file',
                        dest='test_file',
                        action='store',
                        required=True,
                        help='path to the test dataset.')
    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    # check if train and test file are exist and are .csv files.
    if not os.path.exists(train_file):
        raise FileNotFoundError('Training file does not exist! :{}'.format(train_file))
    if not os.path.exists(test_file):
        raise FileNotFoundError('Test file does not exist! :{}'.format(test_file))
    if not re.match('.*.csv', train_file):
        raise TypeError('Training file is not .csv type! :{}'.format(train_file))
    if not re.match('.*.csv', test_file):
        raise TypeError('Test file is not .csv type! :{}'.format(test_file))

    # read train and test datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    size_total = len(train_df) + len(test_df)
    logging.info('Total training set size: {} (%100)'.format(size_total))
    logging.info('Size of training set: {} (%{:.2f})'.format(len(train_df), len(train_df)/size_total*100))
    logging.info('Size of test set: {} (%{:.2f})'.format(len(test_df), len(test_df)/size_total*100))

    return train_df, test_df


def feature_engineering(train_df, test_df):
    """

    Converts categorical features into dummy features.

    Parameters
    ----------
    train_df : pandas dataframe
        Training dataset
    test_df : pandas dataframe
        Test dataset

    Returns
    -------
    train_df : pandas dataframe
        Training dataset
    test_df : pandas dataframe
        Test dataset

    """

    # Select features to train and test, convert categorical features to dummy features
    feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

    logging.info('Features used: {}'.format(feature_cols))
    print('Features used: {}'.format(feature_cols))

    train_df = pd.get_dummies(train_df[['Survived'] + feature_cols], columns=['Sex', 'Embarked'])
    test_df = pd.get_dummies(test_df[feature_cols], columns=['Sex', 'Embarked'])

    # Scale features
    train_df['Age'] = scale(train_df['Age'])
    train_df['Fare'] = scale(train_df['Fare'])

    test_df['Age'] = scale(test_df['Age'])
    test_df['Fare'] = scale(test_df['Fare'])

    logging.info('Scaled features')

    # drop nan features
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    return train_df, test_df


def train_and_test(train_df, test_df):
    """

    Train and test using SVM and return predictions.

    Parameters
    ----------
    train_df : pandas dataframe
        Training dataset

    Returns
    -------
    predictions : numpy array
        1D numpy array containing predictions for test set, test_df.

    """
    # set up an SVM classifier
    C = .8
    logging.info('Training using an SVM model.')
    logging.info('SVM Params: C: {}'.format(C))
    clf = svm.SVC(C=C)
    clf.fit(train_df.iloc[:, 1:], train_df['Survived'])
    # create training and test split
    x_train, x_val, y_train, y_val = train_test_split(train_df.iloc[:, 1:], train_df.iloc[:,0], test_size=0.3)#,

    pred_train = clf.predict(x_train)
    pred_val = clf.predict(x_val)
    size_total = len(x_train) + len(x_val) + len(test_df)

    prev_train = sum(y_train)/len(y_train)
    prev_val = sum(y_val)/len(y_val)

    print('prevalence of training: %{:.2f} validation: %{:.2f}'.format(prev_train, prev_val))
    logging.info('prevalence of training dataset: %{:.2f} validation dataset: %{:.2f}'.format(prev_train, prev_val))

    logging.info('Size of training set: {} (%{:.2f})'.format(len(x_train), len(x_train)/size_total*100))
    logging.info('Size of validation set: {} (%{:.2f})'.format(len(x_val), len(x_val)/size_total*100))
    logging.info('Size of test set: {} (%{:.2f})'.format(len(test_df), len(test_df)/size_total*100))

    logging.info('training accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_train, pred_train), precision_score(y_train, pred_train), recall_score(y_train, pred_train)))
    logging.info('validation accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_val, pred_val), precision_score(y_val, pred_val), recall_score(y_val, pred_val)))

    print('training accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_train, pred_train), precision_score(y_train, pred_train), recall_score(y_train, pred_train)))
    print('validation accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_val, pred_val), precision_score(y_val, pred_val), recall_score(y_val, pred_val)))

    tn, fp, fn, tp = confusion_matrix(y_train, pred_train).ravel()
    logging.info('training conf matrix: \n{}'.format(confusion_matrix(y_train, pred_train)))
    logging.info('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    print('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))

    tn, fp, fn, tp = confusion_matrix(y_val, pred_val).ravel()
    logging.info('validation conf matrix: \n{}'.format(confusion_matrix(y_val, pred_val)))
    logging.info('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    print('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))

    logging.info('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    logging.info('Predictions training: \n{}'.format(pred_train))
    logging.info('Predictions validation: \n{}'.format(pred_val))

    plot_roc_curve(clf, x_train, y_train)
    plt.title('Training set ROC Curve')
    plt.savefig('plots/training_ROC.png')
    plt.show()
    plt.clf()
    plot_roc_curve(clf, x_val, y_val)
    plt.title('Validation set ROC Curve')
    plt.savefig('plots/validation_ROC.png')
    plt.show()
    plt.clf()
    return clf


if __name__ == '__main__':
    start = time.time()
    # set up logger
    FORMAT = '%(asctime)-15s %(levelname)s-8s %(message)s'
    logging.basicConfig(filename='logs/run_{}.log'.format(time.strftime("%Y.%m.%d_%H.%M.%S"), time.localtime()), format=FORMAT, level=logging.INFO)

    train_df, test_df = read_data()
    train_df, test_df = feature_engineering(train_df, test_df)
    model = train_and_test(train_df, test_df)

    predictions = model.predict(test_df)
    print('End')
    end = time.time()
    logging.info('Took {:.2f} seconds'.format(end-start))
