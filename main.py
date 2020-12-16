# Fatma Usta - December 2020
import os
import re
import time
import argparse
import logging
import pandas as pd
import sklearn

def main():
    # set up argumet parser
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

    return train_file, test_file

if __name__ == '__main__':
    start = time.time()
    # set up logger
    FORMAT = '%(asctime)-15s %(levelname)s-8s %(message)s'
    logging.basicConfig(filename='run_{}.log'.format(time.strftime("%Y.%m.%d_%H.%M.%S"), time.localtime()), format=FORMAT, level=logging.DEBUG)

    train_file, test_file = main()
    # read train and test datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    size_total = len(train_df) + len(test_df)
    logging.info('Size of training set: {} (%{:.2f})'.format(len(train_df), len(train_df)/size_total*100))
    logging.info('Size of test set: {} (%{:.2f})'.format(len(test_df), len(test_df)/size_total*100))

    # Select target column
    train_y = train_df[['Survived']]

    # Select features to train and test
    train_x = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    test_y = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]


    print('End')
    end = time.time()

