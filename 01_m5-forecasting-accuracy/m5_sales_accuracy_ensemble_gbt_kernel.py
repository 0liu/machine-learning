"""
This script is meant to be copied into multiple kernels on Kaggle, to train multiple models in parallel.

The three processed data files should be uploaded to Kaggle before running kernels.
"""

# Python libs
import warnings
import joblib
from pathlib import Path
from itertools import product
from math import ceil
import random
import numpy as np
import pandas as pd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

# Module settings
mpl.rc("figure", facecolor="white", dpi=196)
pd.set_option('display.max_columns', None)  # show all columns


# Globals / Class attributes
FLAT_INDEX_COLS = ['id', 'd']
TARGET_COL = 'sales'
ID_D_TARGET_COLS = ['id', 'd', TARGET_COL]
END_TRAIN = 1941
SHIFT_DAY = 28

SALES_PRICES_CALENDAR_DATA_FILE = '../input/m5salesaccuracyensemblegb/sales_prices_calendar.pkl'
SALES_LAG_ROLL_STATS_FILE = '../input/m5salesaccuracyensemblegb/sales_lag_roll_stats.pkl'
SALES_AGG_STATS_FILE = '../input/m5salesaccuracyensemblegb/sales_agg_stats.pkl'


IDs_ALL = {
    "store_id": [
        "CA_1",
        "CA_2",
        "CA_3",
        "CA_4",
        "TX_1",
        "TX_2",
        "TX_3",
        "WI_1",
        "WI_2",
        "WI_3",
    ],
    "cat_id": ["HOBBIES", "HOUSEHOLD", "FOODS"],
    "dept_id": [
        "HOBBIES_1",
        "HOBBIES_2",
        "HOUSEHOLD_1",
        "HOUSEHOLD_2",
        "FOODS_1",
        "FOODS_2",
        "FOODS_3",
    ],
}

sales_lag_rolling_stats_cols = [
    "id",
    "d",
    "sales",
    "sales_lag_28",
    "sales_lag_29",
    "sales_lag_30",
    "sales_lag_31",
    "sales_lag_32",
    "sales_lag_33",
    "sales_lag_34",
    "sales_lag_35",
    "sales_lag_36",
    "sales_lag_37",
    "sales_lag_38",
    "sales_lag_39",
    "sales_lag_40",
    "sales_lag_41",
    "sales_lag_42",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "rolling_mean_60",
    "rolling_std_60",
    "rolling_mean_180",
    "rolling_std_180",
    "rolling_mean_shift1_win7",
    "rolling_mean_shift1_win14",
    "rolling_mean_shift1_win30",
    "rolling_mean_shift1_win60",
    "rolling_mean_shift7_win7",
    "rolling_mean_shift7_win14",
    "rolling_mean_shift7_win30",
    "rolling_mean_shift7_win60",
    "rolling_mean_shift14_win7",
    "rolling_mean_shift14_win14",
    "rolling_mean_shift14_win30",
    "rolling_mean_shift14_win60",
]
sales_agg_stats_cols = [
    "id",
    "d",
    "level2_state_id_mean",
    "level2_state_id_std",
    "level3_store_id_mean",
    "level3_store_id_std",
    "level4_cat_id_mean",
    "level4_cat_id_std",
    "level5_dept_id_mean",
    "level5_dept_id_std",
    "level6_state_id_cat_id_mean",
    "level6_state_id_cat_id_std",
    "level7_state_id_dept_id_mean",
    "level7_state_id_dept_id_std",
    "level8_store_id_cat_id_mean",
    "level8_store_id_cat_id_std",
    "level9_store_id_dept_id_mean",
    "level9_store_id_dept_id_std",
    "level10_item_id_mean",
    "level10_item_id_std",
    "level11_item_id_state_id_mean",
    "level11_item_id_state_id_std",
    "level12_item_id_store_id_mean",
    "level12_item_id_store_id_std",
]


loop_cols_L3_store = ['store_id']
loop_cols_L8_store_cat = ['store_id', 'cat_id']
loop_cols_L9_store_dept = ['store_id', 'dept_id']

drop_cols_L3_store = ['state_id', 'store_id']
drop_cols_L8_store_cat = ['state_id', 'store_id', 'cat_id']
drop_cols_L9_store_dept = ['state_id', 'store_id', 'cat_id', 'dept_id']

rec_lag_rolling_cols = sales_lag_rolling_stats_cols[3:]
nonrec_lag_rolling_cols = [f'sales_lag_{i}' for i in range(28, 43)] + [f'rolling_{stat}_{i}' for i in (7, 14, 30, 60, 180) for stat in ('mean', 'std')]

rec_agg_stats_cols_L3_store = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level4', 'level5', 'level10')]
rec_agg_stats_cols_L8_store_cat = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level9', 'level12')]
rec_agg_stats_cols_L9_store_dept = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level12',)]

nonrec_agg_stats_cols_L3_store = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level9', 'level11')]
nonrec_agg_stats_cols_L8_store_cat = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level9', 'level12')]
nonrec_agg_stats_cols_L9_store_dept = [col for col in sales_agg_stats_cols if col.split('_')[0] in ('level12',)]

rec_start_d_L3 = 0
rec_start_d_L8 = 700
rec_start_d_L9 = 700
rec_num_leaves_L3 = 2*11 - 1
rec_num_leaves_L8 = 2**8 - 1
rec_num_leaves_L9 = 2**8 - 1
rec_min_data_in_leaf_L3 = 2**12 - 1
rec_min_data_in_leaf_L8 = 2**8 - 1
rec_min_data_in_leaf_L9 = 2**8 - 1

nonrec_start_d = 710
nonrec_num_leaves = 2**8 - 1
nonrec_min_data_in_leaf = 2**8 - 1

feature_hyperparas = {
    "rec_L3_store": {
        "loop_cols": loop_cols_L3_store,
        "drop_cols": drop_cols_L3_store,
        "lag_rolling_cols": rec_lag_rolling_cols,
        "agg_stats_cols": rec_agg_stats_cols_L3_store,
        "start_d": rec_start_d_L3,
        "num_leaves": rec_num_leaves_L3,
        "min_data_in_leaf": rec_min_data_in_leaf_L3,
    },

    "rec_L8_store_cat": {
        "loop_cols": loop_cols_L8_store_cat,
        "drop_cols": drop_cols_L8_store_cat,
        "lag_rolling_cols": rec_lag_rolling_cols,
        "agg_stats_cols": rec_agg_stats_cols_L8_store_cat,
        "start_d": rec_start_d_L8,
        "num_leaves": rec_num_leaves_L8,
        "min_data_in_leaf": rec_min_data_in_leaf_L8,
    },

    "rec_L9_store_dept": {
        "loop_cols": loop_cols_L9_store_dept,
        "drop_cols": drop_cols_L9_store_dept,
        "lag_rolling_cols": rec_lag_rolling_cols,
        "agg_stats_cols": rec_agg_stats_cols_L9_store_dept,
        "start_d": rec_start_d_L9,
        "num_leaves": rec_num_leaves_L9,
        "min_data_in_leaf": rec_min_data_in_leaf_L9,
    },

    "nonrec_L3_store": {
        "loop_cols": loop_cols_L3_store,
        "drop_cols": drop_cols_L3_store,
        "lag_rolling_cols": nonrec_lag_rolling_cols,
        "agg_stats_cols": nonrec_agg_stats_cols_L3_store,
        "start_d": nonrec_start_d,
        "num_leaves": nonrec_num_leaves,
        "min_data_in_leaf": nonrec_min_data_in_leaf,
    },

    "nonrec_L8_store_cat": {
        "loop_cols": loop_cols_L8_store_cat,
        "drop_cols": drop_cols_L8_store_cat,
        "lag_rolling_cols": nonrec_lag_rolling_cols,
        "agg_stats_cols": nonrec_agg_stats_cols_L8_store_cat,
        "start_d": nonrec_start_d,
        "num_leaves": nonrec_num_leaves,
        "min_data_in_leaf": nonrec_min_data_in_leaf,
    },

    "nonrec_L9_store_dept": {
        "loop_cols": loop_cols_L9_store_dept,
        "drop_cols": drop_cols_L9_store_dept,
        "lag_rolling_cols": nonrec_lag_rolling_cols,
        "agg_stats_cols": nonrec_agg_stats_cols_L9_store_dept,
        "start_d": nonrec_start_d,
        "num_leaves": nonrec_num_leaves,
        "min_data_in_leaf": nonrec_min_data_in_leaf,
    },
}

lgb_hyperparas = {
    "boosting": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "learning_rate": 0.015,
    "num_leaves": 2 ** 8 - 1,
    "min_data_in_leaf": 2 ** 8 - 1,
    "feature_fraction": 0.5,
    "max_bin": 100,
    "num_iterations": 3000,
    "boost_from_average": False,
    "verbosity": -1,  # FATAL, suppress lgb warnings
    "seed": 1995,
    "num_threads": 10
}

def get_data_for_modeling(loop_ids: tuple, hyperparas: dict) -> tuple:
    """
    Read data and Filter / Merge features per aggregation level and IDs.
    """

    global ID_D_TARGET_COLS
    global SALES_PRICES_CALENDAR_DATA_FILE, SALES_LAG_ROLL_STATS_FILE, SALES_AGG_STATS_FILE

    print("Load SALES_PRICES_CALENDAR_DATA_FILE ...")
    data = pd.read_pickle(SALES_PRICES_CALENDAR_DATA_FILE)
    
    # Cut off from starting day
    print("Cut off from starting day ...")
    data = data[data['d'] >= hyperparas['start_d']]
    
    # Filter data by agg level ids
    print("Create agg id filter mask ...")
    id_mask = True
    for loop_col, loop_val in zip(hyperparas['loop_cols'], loop_ids):
        id_mask &= (data[loop_col] == loop_val)
    data = data[id_mask]

    # Drop aggregated id columns
    data = data.drop(hyperparas['drop_cols'], axis=1)

    # Lags / rollings features
    print("Load lag rolling stats ...")
    df_lags_rolling = pd.read_pickle(SALES_LAG_ROLL_STATS_FILE)[hyperparas['lag_rolling_cols']]
    df_lags_rolling = df_lags_rolling[df_lags_rolling.index.isin(data.index)]

    # Agg mean/std features
    print("Load agg stats ...")
    df_agg_mean_std = pd.read_pickle(SALES_AGG_STATS_FILE)[hyperparas['agg_stats_cols']]
    df_agg_mean_std = df_agg_mean_std[df_agg_mean_std.index.isin(data.index)]

    # Merge data
    print("Merge data with stats ...")
    data = pd.concat([data, df_lags_rolling, df_agg_mean_std], axis=1).reset_index(drop=True)
    features = [col for col in data.columns if col not in ID_D_TARGET_COLS]

    return data, features


def get_train_val_test_data(loop_ids, hyperparas):
    global END_TRAIN, SHIFT_DAY, TARGET_COL, FLAT_INDEX_COLS
    
    data, features = get_data_for_modeling(loop_ids, hyperparas)

    train_mask = data['d'] <= END_TRAIN
    valid_mask = (data['d'] > END_TRAIN - SHIFT_DAY) & train_mask
    test_mask = ~train_mask

    train = lgb.Dataset(data[train_mask][features], label=data[train_mask][TARGET_COL])

    X_valid = data[valid_mask][features]
    valid = lgb.Dataset(X_valid, label=data[valid_mask][TARGET_COL])
    id_valid = data[valid_mask][FLAT_INDEX_COLS]
    id_valid['id'] = id_valid['id'].str.replace('evaluation', 'validation')

    test = data[test_mask]
    X_test = test[features]
    id_test = test[FLAT_INDEX_COLS]

    return train, valid, X_valid, id_valid, X_test, id_test

def train_one_model(train, valid, hyperparas: dict, lgb_hyperparas=lgb_hyperparas) -> None:
    """
    At a particular aggregation level, train one model for a particular id group (e.g. store_id and cat_id).
    """
    # Update parameters
    lgb_hyperparas = lgb_hyperparas.copy()
    lgb_hyperparas['num_leaves'] = hyperparas['num_leaves']
    lgb_hyperparas['min_data_in_leaf'] = hyperparas['min_data_in_leaf']

    # Train
    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    model = lgb.train(lgb_hyperparas, train, valid_sets=[valid, train], callbacks=[lgb.log_evaluation(period=100)])
    warnings.filterwarnings('default')

    # Feature importace
    featimp = pd.DataFrame({'feature': model.feature_name(), 'importance': model.feature_importance()}).sort_values('importance', ascending=False)

    return model, featimp

def predict_one_model(X_test: pd.DataFrame, id_test: pd.DataFrame, model):
    global TARGET_COL
    
    id_test[TARGET_COL] = model.predict(X_test)
    return id_test

def train_predict_one_level(agg_level, feature_hyperparas=feature_hyperparas, warm_start_loop_ids=None, warm_start_label=''):
    global IDs_ALL
    hyperparas = feature_hyperparas[agg_level]
    if warm_start_loop_ids is None:  # run all ID groups
        loop_ids_all = list(product(*(IDs_ALL[col] for col in hyperparas['loop_cols'])))
    else:
        loop_ids_all = warm_start_loop_ids
    results = {}

    for loop_ids in loop_ids_all:
        print(f'\n=== Train / Predict {agg_level} - {loop_ids} ===')

        print("Get data and split to Train / Validation / Test ...")
        train, valid, X_valid, id_valid, X_test, id_test = get_train_val_test_data(loop_ids, hyperparas)

        print("Train model ...")
        model, featimp = train_one_model(train, valid, hyperparas)

        print("Predict ...")
        val_pred = predict_one_model(X_valid, id_valid, model)
        test_pred = predict_one_model(X_test, id_test, model)

        # Save results
        print("Dump model and results ...")
        loop_ids_str = '_'.join(list(loop_ids))
        results[loop_ids_str] = {
            'model': model,
            'feature_importance': featimp,
            'val_pred': val_pred,
            'test_pred': test_pred,
        }
        joblib.dump(results, f'{agg_level}{warm_start_label}.pkl')  # overwrite at each iteration

    
L3_loop_ids = list(product(*(IDs_ALL[col] for col in loop_cols_L3_store)))
L8_loop_ids = list(product(*(IDs_ALL[col] for col in loop_cols_L8_store_cat)))
L9_loop_ids = list(product(*(IDs_ALL[col] for col in loop_cols_L9_store_dept)))

# Split each level to 2 parts so that it can finish running within Kaggle's 9-hour limit.
L3_loop_ids_A, L3_loop_ids_B = L3_loop_ids[: len(L3_loop_ids)//2], L3_loop_ids[len(L3_loop_ids)//2 :]
L8_loop_ids_A, L8_loop_ids_B = L8_loop_ids[: len(L8_loop_ids)//2], L8_loop_ids[len(L8_loop_ids)//2 :]
L9_loop_ids_A, L9_loop_ids_B = L9_loop_ids[: len(L9_loop_ids)//2], L9_loop_ids[len(L9_loop_ids)//2 :]

# Uncomment one of lines below to train a part of models
# train_predict_one_level('rec_L3_store')
# train_predict_one_level('rec_L8_store_cat')
# train_predict_one_level('rec_L9_store_dept')
# train_predict_one_level('nonrec_L3_store', warm_start_loop_ids=L3_loop_ids_A, warm_start_label='_A')
# train_predict_one_level('nonrec_L8_store_cat', warm_start_loop_ids=L8_loop_ids_A, warm_start_label='_A')
# train_predict_one_level('nonrec_L9_store_dept', warm_start_loop_ids=L9_loop_ids_A, warm_start_label='_A')