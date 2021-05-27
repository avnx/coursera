import numpy as np
import pandas as pd
from itertools import product

from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error as mse


def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def fill_with_0_target(df):
    index_cols = ['shop_id', 'item_id', 'date_block_num']

    # For every month we create a grid from all shops/items combinations from that month
    grid = [] 
    for block_num in df['date_block_num'].unique():
        cur_shops = df[df['date_block_num']==block_num]['shop_id'].unique()
        cur_items = df[df['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
    gb = df.groupby(index_cols,as_index=False).agg({'item_cnt_day': 'sum'}) \
                                          .rename(columns={'item_cnt_day': 'target'})

    all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
    #sort the data
    all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
    return all_data


def get_split_points(data):
    """
    get split points for train/val in CV5
    """
    train_split_points, val_split_points = [], []
    min_block_num = data.date_block_num.min()
    max_block_num = data.date_block_num.max()
    
    for date_block_num in range(min_block_num, min_block_num + 5):
        begin_val = data.date_block_num.searchsorted(max_block_num - 4 + date_block_num - min_block_num)
        begin_train = 0 #data.date_block_num.searchsorted(date_block_num)
        end_train = begin_val
        if date_block_num == min_block_num + 4:
            end_val = len(data)
        else:
            end_val = data.date_block_num.searchsorted(max_block_num - 4 + date_block_num - min_block_num + 1)
        val_split_points.append((begin_val, end_val))
        train_split_points.append((begin_train, end_train))
    return train_split_points, val_split_points
        

def get_train_val(data, train_split_points, val_split_points):
    for train_points, val_points in zip(train_split_points, val_split_points):
        train = data[train_points[0]:train_points[1]]
        val = data[val_points[0]:val_points[1]]
        yield train, val

        
def add_shop_last_stat(data):
    grouped = data.groupby(['date_block_num', 'shop_id']).target.sum()
    grouped = grouped.reset_index()
    grouped['date_block_num'] += 1
    grouped.rename(columns={'target': 'prev_shop_sales'}, inplace=True)
    return pd.merge(data, grouped,how='left',on=['date_block_num', 'shop_id'])
    

def add_item_last_stat(data):
    grouped = data.groupby(['date_block_num', 'item_id']).target.sum()
    grouped = grouped.reset_index()
    grouped['date_block_num'] += 1
    grouped.rename(columns={'target': 'prev_item_sales'}, inplace=True)
    return pd.merge(data, grouped,how='left',on=['date_block_num', 'item_id'])


def get_new_val(val, extra_items, extra_idces):
    """
    validation with equal portion of new objects in each split
    """
    if len(extra_idces) / len(val) < 0.071:
        new_n_old_objects = int(len(extra_idces) / 0.071 - len(extra_idces))
        new_val = val.loc[~val['item_id'].isin(extra_items)]
        drop_indices = np.random.choice(new_val.index, len(new_val) - new_n_old_objects, replace=False)
        new_val = val.drop(drop_indices)
    else:
        n_old_objects = len(val.loc[~val['item_id'].isin(extra_items)])
        new_n_new_objects = int(n_old_objects * 0.071 / (1 - 0.071 ))
        new_val = val.loc[val['item_id'].isin(extra_items)]
        drop_indices = np.random.choice(new_val.index, len(new_val) - new_n_new_objects, replace=False)
        new_val = val.drop(drop_indices)
    return new_val


def fit_and_eval(data, algo='rfr', early_stopping_rounds=5):
    """
    fit either Random Forest or Catboost using CV to evaluate quality
    """
    train_split_points, val_split_points = get_split_points(data)

    val_results = []
    best_iterations = []
    for train, val in get_train_val(data, train_split_points, val_split_points):
        extra_items = set(val.item_id) - set(train.item_id)
        extra_idces = val.loc[val['item_id'].isin(extra_items)].index - val.index.min()
        val = get_new_val(val, extra_items, extra_idces).reset_index(drop=True)
        X_train = np.array(train.drop(['date_block_num', 'target', 'item_id', 'shop_id'], axis=1))
        X_val = np.array(val.drop(['date_block_num', 'target', 'item_id', 'shop_id'], axis=1))
        y_train, y_val = np.array(train['target']), np.array(val['target'])
        if algo == 'rfr':
            model = RandomForestRegressor(n_estimators=64, n_jobs=-1)
            model.fit(X_train, y_train)
        else:
            model = CatBoostRegressor()
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds, silent=True)
        preds = model.predict(X_val)
        extra_idces = val.loc[val['item_id'].isin(extra_items)].index - val.index.min()
        preds[extra_idces] += 0.3
        val_results.append(mse(preds, y_val, squared=False))
        best_iterations.append(model.best_iteration_)
    print('Model: ', algo)
    print('feat importancies: ', model.feature_importances_)
    print('rmse on val: ', val_results)
    print('best iterations: ', best_iterations)
    return model
