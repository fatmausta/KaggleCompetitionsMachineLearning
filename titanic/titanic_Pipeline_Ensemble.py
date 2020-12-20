# Fatma Usta - December 2020
import os
import re
import time
import argparse
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)
        return self.clf.fit(x, y).feature_importances_



def set_parameters():
    # Put in our parameters for said classifiers
    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }

    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
    }

    return rf_params, et_params, ada_params, gb_params, svc_params


def create_models(rf_params, et_params, ada_params, gb_params, svc_params, SEED=0):
    # Create 5 objects that represent our 4 models
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

    return rf, et, ada, gb, svc


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


def select_features(train_df, test_df):
    """

    Train and test using SVM and return predictions.

    Parameters
    ----------
    train_df : pandas dataframe
        Training dataset

    Returns
    -------
    x_train, y_train, x_test


    """

    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train_df = train_df.drop(drop_elements + ['CategoricalAge', 'CategoricalFare'], axis=1)

    # create training and test split
    x_train = train_df.drop(['Survived'], axis=1)
    y_train = train_df['Survived']
    x_test = test_df.drop(drop_elements, axis=1)

    return x_train, y_train, x_test


def train_and_test(x_train, y_train):
    # set up an SVM classifier
    C = .8
    logging.info('Training using an SVM model.')
    logging.info('SVM Params: C: {}'.format(C))
    logging.info('Features Used: {}'.format(x_train.keys()))
    print('Features Used: {}'.format(x_train.keys()))
    clf = SVC(C=C)
    clf.fit(x_train, y_train)
    return clf


def predict_and_log(clf, x_train, y_train, x_test):
    """

     predictions : numpy array
        1D numpy array containing predictions for test set, test_df.

    :param clf:
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)

    size_total = len(x_train) + len(x_test)

    prev_train = sum(y_train)/len(y_train)

    logging.info('Size of training set: {} (%{:.2f})'.format(len(x_train), len(x_train)/size_total*100))
    logging.info('Size of test set: {} (%{:.2f})'.format(len(test_df), len(test_df)/size_total*100))

    print('prevalence of training: %{:.2f}'.format(prev_train))
    logging.info('prevalence of training dataset: %{:.2f}'.format(prev_train))

    logging.info('training accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_train, pred_train), precision_score(y_train, pred_train), recall_score(y_train, pred_train)))

    print('training accuracy: %{:.2f} precision: %{:.2f} recall: %{:.2f}'.format(accuracy_score(y_train, pred_train), precision_score(y_train, pred_train), recall_score(y_train, pred_train)))

    tn, fp, fn, tp = confusion_matrix(y_train, pred_train).ravel()
    logging.info('training conf matrix: \n{}'.format(confusion_matrix(y_train, pred_train)))
    logging.info('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))
    print('tn: {}, fp: {}, fn: {}, tp:{}'.format(tn, fp, fn, tp))

    logging.info('Predictions training: \n{}'.format(pred_train))

    plt.clf()
    plot_roc_curve(clf, x_train, y_train)
    plt.title('Training set ROC Curve')
    plt.savefig('plots/training_ROC.png')

    return pred_test


def print_log_results(y_true, y_pred, dataset_name):
    print('{} accuracy: {}'.format(dataset_name, accuracy_score(y_true, y_pred)))
    print('{} precision: {}'.format(dataset_name, precision_score(y_true, y_pred)))
    print('{} recall: {}'.format(dataset_name, recall_score(y_true, y_pred)))

    logging.info('{} accuracy: {}'.format(dataset_name, accuracy_score(y_true, y_pred)))
    logging.info('{} precision: {}'.format(dataset_name, precision_score(y_true, y_pred)))
    logging.info('{} recall: {}'.format(dataset_name, recall_score(y_true, y_pred)))


def aggregate_predictions_xgboost(x_train, y_train, x_val, y_val, x_test):

    eval_set = [(x_train, y_train), (x_val, y_val)]
    gbm = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=50,
        max_depth=4,
        min_child_weight=10,
        # gamma=1,
        gamma=0.8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(x_train, y_train, early_stopping_rounds=10, eval_set=eval_set,
                                eval_metric=["error", "logloss"], verbose=True)

    pred_train = gbm.predict(x_train)
    pred_val = gbm.predict(x_val)
    pred_test = gbm.predict(x_test)

    print_log_results(y_train, pred_train, 'training')
    print_log_results(y_val, pred_val, 'validation')

    # retrieve performance metrics
    results = gbm.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig('plots/logloss.png')
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig('plots/error.png')

    return pred_train, pred_val, pred_test


def aggregate_predictions_voting(x_train, y_train, x_val, y_val, x_test):
    """
    x_train, x_val, x_test are all predictions.
    This function votes the results.

    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param x_test:
    :return:
    """

    pred_train = x_train.mode(axis=1).values
    pred_val = x_val.mode(axis=1).values
    pred_test = x_test.mode(axis=1).values

    print_log_results(y_train, pred_train, 'training')
    print_log_results(y_val, pred_val, 'validation')

    return pred_train, pred_val, pred_test


if __name__ == '__main__':
    start = time.time()
    # set up logger
    FORMAT = '%(asctime)-15s %(levelname)s-8s %(message)s'
    logging.basicConfig(filename='logs/run_{}.log'.format(time.strftime("%Y.%m.%d_%H.%M.%S"), time.localtime()), format=FORMAT, level=logging.INFO)

    train_df, test_df = read_data()
    train_df, test_df = feature_engineering(train_df, test_df)
    x_train, y_train, x_test = select_features(train_df, test_df)

    rf_params, et_params, ada_params, gb_params, svc_params = set_parameters()
    rf, et, ada, gb, svc = create_models(rf_params, et_params, ada_params, gb_params, svc_params)

    seed = 7
    val_size = 0.33
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)

    training_predictions = {}
    validation_predictions = {}
    test_predictions = {}
    feature_importances = {'features': x_train.keys()}
    for name, model in zip(['rf', 'et', 'ada', 'gb', 'svc'], [rf, et, ada, gb, svc]):
        model.fit(x_train, y_train)
        training_predictions[name] = model.predict(x_train)
        validation_predictions[name] = model.predict(x_val)
        test_predictions[name] = model.predict(x_test)
        if name!='svc':
            feature_importances[name] = model.feature_importances(x_train, y_train)

    # save feature importances
    pd.DataFrame(feature_importances).to_csv('feature_importances.csv')
    logging.info('Feature_importances')
    logging.info(feature_importances)
    # use predictions and features as features
    # x_train = pd.concat([x_train, pd.DataFrame(training_predictions, ignore_index=True)], axis=1)
    # x_val = pd.concat([x_val, pd.DataFrame(validation_predictions)], axis=1)
    # x_test = pd.concat([x_test, pd.DataFrame(test_predictions)], axis=1)

    # use predictions as features
    x_train = pd.DataFrame(training_predictions)
    x_val = pd.DataFrame(validation_predictions)
    x_test = pd.DataFrame(test_predictions)

    # pred_train, pred_val, pred_test = aggregate_predictions_xgboost(x_train, y_train, x_val, y_val, x_test)
    pred_train, pred_val, pred_test = aggregate_predictions_voting(x_train, y_train, x_val, y_val, x_test)

    submission = {'PassengerId': test_df['PassengerId'], 'Survived': pred_test}
    df = pd.DataFrame(submission)
    df.set_index('PassengerId', inplace=True)
    df.to_csv('submission_titanic.csv')

    print('End')
    end = time.time()
    logging.info('Took {:.2f} seconds'.format(end-start))
