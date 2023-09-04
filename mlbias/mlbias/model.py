"""
Code to train and evaluate the model.
"""
import pandas as pd
from loguru import logger
import numpy as np


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