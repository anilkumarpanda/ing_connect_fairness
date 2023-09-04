"""
Code to train and evaluate the model.
"""
import pandas as pd
from loguru import logger
import numpy as np
import xgboost as xgb

def get_monotone_constraints(data, target, corr_threshold):
    """Calculate monotonic constraints.
    Monotonic constraints help in explainability.
    It enforces the model to maintain certain relationship.

    Using a cutoff on Spearman correlation between features and target,
    return a tuple ready to pass into XGBoost.

    Spearman correlation is nice because it considers monotonicity rather than
    linearity (as is the case with Pearson correlation coefficient).


    Args:
        data (pd.DataFrame): A DataFrame containing the features in the order they appear to XGBoost, as well as the target variable.
        target (str): The name of the column with the target variable in 'data'.
        corr_threshold (float): The Spearman correlation threshold.

    Returns:
        tuple: A tuple with values in {-1, 0, 1}, where each element corresponds to a column in data (excluding the target itself). Ready to pass into xgb.train()

    """

    corr = pd.Series(data.corr(method='spearman')[target]).drop(target)
    monotone_constraints = tuple(np.where(corr < -corr_threshold,
                                          -1,
                                          np.where(corr > corr_threshold,
                                                   1,
                                                   0)))
    return monotone_constraints

def train_model(data: pd.DataFrame):
    """
    Train the model.
    """
    logger.info('Reading in data')
    data = pd.read_csv(data)
    data['SEX'] = np.where(data['SEX'] == 1, 'male', 'female')
    race_map = {1: 'hispanic', 2: 'black', 3: 'white', 4: 'asian'}
    data['RACE'] = data['RACE'].apply(lambda x: race_map[x])
    
    # Modify the data so there is a distributional difference
    # between borrowers of different race/ethnicities.
    logger.info('Modifying data')

    new_limit_bal = data['LIMIT_BAL'] - 20000*np.random.randn(len(data))
    new_limit_bal[new_limit_bal <= 10000] = 10000
    data['LIMIT_BAL'] = np.where((data['RACE'] == 'hispanic') | (data['RACE'] == 'black'),
                                new_limit_bal,
                                data['LIMIT_BAL'])

    for i in range(1, 7):
        delta = 1000*np.random.randn(len(data))
        new_pay = data[f'PAY_AMT{i}'] - delta
        new_pay[new_pay < 0] = 0

        new_bill = data[f'BILL_AMT{i}'] - delta
        new_bill[new_bill < 0] = 0

        data[f'PAY_AMT{i}'] = np.where((data['RACE'] == 'hispanic') | (data['RACE'] == 'black'),
                                    new_pay,
                                    data[f'PAY_AMT{i}'])
        data[f'BILL_AMT{i}'] = np.where((data['RACE'] == 'hispanic') | (data['RACE'] == 'black'),
                                        new_bill,
                                        data[f'BILL_AMT{i}'])
    
    # Split the data into train validation and test
    seed = 12345
    np.random.seed(seed)

    split_train_test = 2/3

    split = np.random.rand(len(data)) < split_train_test
    train = data[split].copy()
    test = data[~split].copy()

    split_test_valid = 1/2

    split = np.random.rand(len(test)) < split_test_valid
    valid = test[split].copy()
    test = test[~split].copy()

    del data

    logger.info(f"Train/Validation/Test sizes: {len(train)}/{len(valid)}/{len(test)}")

    id_col = 'ID'
    groups = ['SEX', 'RACE', 'EDUCATION', 'MARRIAGE', 'AGE']
    target = 'DELINQ_NEXT'
    features = [col for col in train.columns if col not in groups + [id_col, target]]

    dtrain = xgb.DMatrix(train[features],
                        label=train[target])

    dvalid = xgb.DMatrix(valid[features],
                        label=valid[target])
    
    # Calculate monotonic constraints
    correlation_cutoff = 0.1

    monotone_constraints = get_monotone_constraints(train[features+[target]],
                                                target,
                                                correlation_cutoff)
    
    # Feed the model the global bias
    # refers to the initial prediction value assigned to all
    # instances before the boosting process begins.
    # It acts as a starting point for the gradient boosting algorithm.

    base_score = train[target].mean()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 1.0,
        'max_depth': 5,
        'base_score': base_score,
        'monotone_constraints': dict(zip(features, monotone_constraints)),
        'seed': seed
    }

    # Train using early stopping on the validation dataset.
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    logger.info('Training model')
    model_constrained = xgb.train(params,
                                dtrain,
                                num_boost_round=200,
                                evals=watchlist,
                                early_stopping_rounds=10,
                                verbose_eval=False)

    train[f'p_{target}'] = model_constrained.predict(dtrain)
    valid[f'p_{target}'] = model_constrained.predict(dvalid)
    test[f'p_{target}'] = model_constrained.predict(xgb.DMatrix(test[features], label=test[target]))    
    logger.info('Model trained')
    return train, valid, test


def perf_metrics(y_true, y_score, pos=1, neg=0, res=0.01):
    """
    Calculates precision, recall, and f1 given outcomes and probabilities.

    Args:
        y_true: Array of binary outcomes
        y_score: Array of assigned probabilities.
        pos: Primary target value, default 1.
        neg: Secondary target value, default 0.
        res: Resolution by which to loop through cutoffs, default 0.01.

    Returns:
        Pandas dataframe of precision, recall, and f1 values.
    """

    eps = 1e-20 # for safe numerical operations

    # init p-r roc frame
    prauc_frame = pd.DataFrame(columns=['cutoff', 'recall', 'precision', 'f1'])

    # loop through cutoffs to create p-r roc frame
    for cutoff in np.arange(0, 1 + res, res):

        # binarize decision to create confusion matrix values
        decisions = np.where(y_score > cutoff , 1, 0)

        # calculate confusion matrix values
        tp = np.sum((decisions == pos) & (y_true == pos))
        fp = np.sum((decisions == pos) & (y_true == neg))
        tn = np.sum((decisions == neg) & (y_true == neg))
        fn = np.sum((decisions == neg) & (y_true == pos))

        # calculate precision, recall, and f1
        recall = (tp + eps)/((tp + fn) + eps)
        precision = (tp + eps)/((tp + fp) + eps)
        f1 = 2/((1/(recall + eps)) + (1/(precision + eps)))


        # add new values to frame
        prauc_frame = prauc_frame.append({'cutoff': cutoff,
                                          'recall': recall,
                                          'precision': precision,
                                          'f1': f1},
                                          ignore_index=True)

    return prauc_frame
