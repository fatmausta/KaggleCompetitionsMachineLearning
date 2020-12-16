# Fatma Usta - December 2020
import os
import re
import time
import argparse
import logging
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score


def main():
    """

    Reads training and test files ad returns them as dataframe.

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
    train_df = pd.get_dummies(train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']], columns=['Sex', 'Embarked'])
    test_df = pd.get_dummies(test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']], columns=['Sex', 'Embarked'])
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
    test_df : pandas dataframe
        Test dataset

    Returns
    -------
    predictions : numpy array
        1D numpy array containing predictions for test set, test_df.

    """
    # set up an SVM classifier
    clf = svm.SVC()
    clf.fit(train_df.iloc[:, 1:], train_df['Survived'])
    predictions = clf.predict(test_df)
    logging.info('Training usign an SVM model.')
    print(predictions)
    logging.info('predictions:\n{}'.format(predictions))
    return predictions


if __name__ == '__main__':
    start = time.time()
    # set up logger
    FORMAT = '%(asctime)-15s %(levelname)s-8s %(message)s'
    logging.basicConfig(filename='run_{}.log'.format(time.strftime("%Y.%m.%d_%H.%M.%S"), time.localtime()), format=FORMAT, level=logging.DEBUG)

    train_df, test_df = main()
    train_df, test_df = feature_engineering(train_df, test_df)
    predictions = train_and_test(train_df, test_df)

    print('End')
    end = time.time()
    logging.info('Took {:.2f} seconds'.format(end-start))
