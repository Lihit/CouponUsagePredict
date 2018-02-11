import pandas as pd
import xgboost as xgb
import os
from sklearn.preprocessing import MinMaxScaler

def XgboostModelPredict(xgbModelPath,ResultPath):
    model = xgb.Booster(model_file=xgbModelPath)
    dataset3 = pd.read_csv('data/DataFeatures/dataset3.csv')
    dataset3.drop_duplicates(inplace=True)
    dataset3_preds = dataset3[['user_id', 'coupon_id', 'date_received']]
    dataset3_x = dataset3.drop(['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)
    dataset3 = xgb.DMatrix(dataset3_x)
    # predict test set
    dataset3_preds['label'] = model.predict(dataset3)
    dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
    dataset3_preds.sort_values(by=['coupon_id', 'label'], inplace=True)
    dataset3_preds.to_csv(os.path.join(ResultPath,"xgb_predict_iter5000.csv"), index=None, header=None)
    print(dataset3_preds.describe())
    # save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open(os.path.join(ResultPath,'xgb_feature_score_iter5000.csv'), 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

if __name__ == '__main__':
    modelPath = 'model'
    modelName = 'xgb2.model'
    ResultPath = 'result'
    xgbModelPath = os.path.join(modelPath,modelName)
    XgboostModelPredict(xgbModelPath,ResultPath)