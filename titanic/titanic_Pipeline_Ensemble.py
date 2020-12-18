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
import numpy as np


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


def feature_engineering(train, test):
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

    full_data = [train, test]

    # Some features of my own that I have added in

    # Feature that tells whether a passenger had a cabin on the Titanic
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Feature engineering steps taken from Sina
    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    # Create a New feature CategoricalAge
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)

    # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    train['FamilySize'] = scale(train['FamilySize'])
    test['FamilySize'] = scale(test['FamilySize'])

    train['Fare'] = scale(train['Fare'])
    test['Fare'] = scale(test['Fare'])

    train['Age'] = scale(train['Age'])
    test['Age'] = scale(test['Age'])

    return train, test


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

    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train_df = train_df.drop(drop_elements + ['CategoricalAge', 'CategoricalFare'], axis=1)

    # create training and test split
    x_train, x_val, y_train, y_val = train_test_split(train_df.drop(['Survived'], axis=1), train_df['Survived'], test_size=0.3)
    x_test = test_df.drop(drop_elements, axis=1)

    # set up an SVM classifier
    C = .8
    logging.info('Training using an SVM model.')
    logging.info('SVM Params: C: {}'.format(C))
    logging.info('Features Used: {}'.format(x_train.keys()))
    print('Features Used: {}'.format(x_train.keys()))
    clf = svm.SVC(C=C)
    clf.fit(x_train, y_train)

    pred_train = clf.predict(x_train)
    pred_val = clf.predict(x_val)
    pred_test = clf.predict(x_test)

    size_total = len(x_train) + len(x_val) + len(x_test)

    prev_train = sum(y_train)/len(y_train)
    prev_val = sum(y_val)/len(y_val)

    logging.info('Size of training set: {} (%{:.2f})'.format(len(x_train), len(x_train)/size_total*100))
    logging.info('Size of validation set: {} (%{:.2f})'.format(len(x_val), len(x_val)/size_total*100))
    logging.info('Size of test set: {} (%{:.2f})'.format(len(test_df), len(test_df)/size_total*100))

    print('prevalence of training: %{:.2f} validation: %{:.2f}'.format(prev_train, prev_val))
    logging.info('prevalence of training dataset: %{:.2f} validation dataset: %{:.2f}'.format(prev_train, prev_val))

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

    plt.clf()
    plot_roc_curve(clf, x_train, y_train)
    plt.title('Training set ROC Curve')
    plt.savefig('plots/training_ROC.png')
    plt.clf()
    plot_roc_curve(clf, x_val, y_val)
    plt.title('Validation set ROC Curve')
    plt.savefig('plots/validation_ROC.png')

    return pred_test


if __name__ == '__main__':
    start = time.time()
    # set up logger
    FORMAT = '%(asctime)-15s %(levelname)s-8s %(message)s'
    logging.basicConfig(filename='logs/run_{}.log'.format(time.strftime("%Y.%m.%d_%H.%M.%S"), time.localtime()), format=FORMAT, level=logging.INFO)

    train_df, test_df = read_data()
    train_df, test_df = feature_engineering(train_df, test_df)
    predictions_test = train_and_test(train_df, test_df)
    logging.info('Predictions testing: \n{}'.format(predictions_test))
    submission = {'PassengerId': test_df['PassengerId'], 'Survived': predictions_test}
    df = pd.DataFrame(submission)
    df.set_index('PassengerId', inplace=True)
    df.to_csv('submission_titanic.csv')

    print('End')
    end = time.time()
    logging.info('Took {:.2f} seconds'.format(end-start))
