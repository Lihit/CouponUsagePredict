import pandas as pd
import xgboost as xgb
import os

def XgboostModelTrain(xgbModelPath):
    dataset1 = pd.read_csv('data/DataFeatures/dataset1.csv')
    dataset1.label.replace(-1, 0, inplace=True)
    dataset2 = pd.read_csv('data/DataFeatures/dataset2.csv')
    dataset2.label.replace(-1, 0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)

    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    dataset1_y = dataset1.label
    dataset1_x = dataset1.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'],
                               axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
    dataset2_y = dataset2.label
    dataset2_x = dataset2.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis=1)
    dataset12_y = dataset12.label
    dataset12_x = dataset12.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis=1)

    dataset1 = xgb.DMatrix(dataset1_x, label=dataset1_y)
    dataset2 = xgb.DMatrix(dataset2_x, label=dataset2_y)
    dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)

    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

    # train on dataset1, evaluate on dataset2
    # watchlist = [(dataset1,'train'),(dataset2,'val')]
    # model = xgb_3500.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)

    watchlist = [(dataset12, 'train')]
    model = xgb.train(params, dataset12, num_boost_round=5000, evals=watchlist)
    model.save_model(xgbModelPath)

if __name__ == '__main__':
    modelPath = 'model'
    modelName = 'xgb2.model'
    xgbModelPath = os.path.join(modelPath,modelName)
    XgboostModelTrain(xgbModelPath)
