import gc 
import os  
import time  
import warnings  
from itertools import combinations
from warnings import simplefilter

import joblib 
import lightgbm as lgb 
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import KFold, TimeSeriesSplit 

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import gc

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


df = pd.read_csv('optiver-trading-at-the-close/train.csv')
df = df.dropna(subset=['target'])
df.reset_index(drop = True, inplace = True)
df_shape = df.shape

def imbalance_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    df['volume'] = df.eval('ask_size + bid_size')
    df['mid_price'] = df.eval('(ask_price + bid_price) / 2')
    df['liquidity_imbalance'] = df.eval('(bid_size-ask_size) / (bid_size+ask_size)')
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})") 

    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])

    return df.replace([np.inf, -np.inf], 0)

def gen_features_two(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    df["minute"] = df["seconds_in_bucket"] // 60  # Minutes

    for key, value in df.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    
    # Generate time and stock-related features
    df = gen_features_two(df)
    gc.collect()  # Perform garbage collection to free up memory
    
    # Select and return the generated features
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_name]

df_train_feats = generate_all_features(df)


lgb_params = {
    "objective": "mae",
    "n_estimators": 5000,
    "num_leaves": 256,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.00871,
    "n_jobs": 4,
    "device": "gpu",
    "verbosity": -1,
    "importance_type": "gain",
}
feature_name = list(df.columns)
print(f"Feature length = {len(feature_name)}")

num_folds = 5
fold_size = 480 // num_folds
gap = 5

models = []
scores = []

model_save_path = 'modelitos_para_despues'  # Directory to save models
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# We need to use the date_id from df_train to split the data
date_ids = df_train_feats['date_id'].values

for i in range(num_folds):
    start = i * fold_size
    end = start + fold_size

    purged_before_start = start - 2
    purged_before_end = start + 2
    purged_after_start = end - 2
    purged_after_end = end + 2
    
    purged_set = ((date_ids >= purged_before_start) & (date_ids <= purged_before_end)) | \
                 ((date_ids >= purged_after_start) & (date_ids <= purged_after_end))
    
    test_indices = (date_ids >= start) & (date_ids < end) & ~purged_set
    train_indices = ~test_indices & ~purged_set
    
    df_fold_train = df_train_feats[train_indices]
    df_fold_train_target = df['target'][train_indices]
    df_fold_valid = df_train_feats[test_indices]
    df_fold_valid_target = df['target'][test_indices]

    print(f"Fold {i+1} Model Training")
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        df_fold_train[feature_name],
        df_fold_train_target,
        eval_set=[(df_fold_valid[feature_name], df_fold_valid_target)],
        callbacks=[
            lgb.callback.early_stopping(stopping_rounds=100),
            lgb.callback.log_evaluation(period=100),
        ],
    )

    models.append(lgb_model)
    model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
    lgb_model.booster_.save_model(model_filename)
    print(f"Model for fold {i+1} saved to {model_filename}")

    fold_predictions = lgb_model.predict(df_fold_valid[feature_name])
    fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
    scores.append(fold_score)

    print(f"Fold {i+1} MAE: {fold_score}")


    del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
    gc.collect()

average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

final_model_params = lgb_params.copy()
final_model_params['n_estimators'] = average_best_iteration

print(f"Training final model with average best iteration: {average_best_iteration}")

final_model = lgb.LGBMRegressor(**final_model_params)
final_model.fit(
    df_train_feats[feature_name],
    df['target'],
    callbacks=[
        lgb.callback.log_evaluation(period=100),
    ],
)

models.append(final_model)

final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
final_model.booster_.save_model(final_model_filename)
print(f"Final model saved to {final_model_filename}")

print(f"Average MAE across all folds: {np.mean(scores)}")
