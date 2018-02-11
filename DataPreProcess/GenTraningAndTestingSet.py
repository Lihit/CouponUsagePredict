import pandas as pd
import numpy as np
from datetime import date
import os
import re

def get_label(s):
    s = s.split(':')
    if s[0]=='null':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1

def GenTraningSet(FeaturePath,datasetIndex):
    TraingSetPath = os.path.join(FeaturePath,'dataset'+str(datasetIndex)+'.csv')
    if os.path.exists(TraingSetPath):
        print("%s has existed!" % TraingSetPath)
        return
    coupon = pd.read_csv(os.path.join(FeaturePath,"coupon"+str(datasetIndex)+"_feature.csv"))
    merchant = pd.read_csv(os.path.join(FeaturePath,"merchant"+str(datasetIndex)+"_feature.csv"))
    user = pd.read_csv(os.path.join(FeaturePath,"user"+str(datasetIndex)+"_feature.csv"))
    user_merchant = pd.read_csv(os.path.join(FeaturePath,"user_merchant"+str(datasetIndex)+".csv"))
    other_feature = pd.read_csv(os.path.join(FeaturePath,"other_feature"+str(datasetIndex)+".csv"))
    dataset = pd.merge(coupon, merchant, on='merchant_id', how='left')
    dataset = pd.merge(dataset, user, on='user_id', how='left')
    dataset = pd.merge(dataset, user_merchant, on=['user_id', 'merchant_id'], how='left')
    dataset = pd.merge(dataset, other_feature, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset.drop_duplicates(inplace=True)

    dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan, 0)
    dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan, 0)
    dataset.user_merchant_received = dataset.user_merchant_received.replace(np.nan, 0)
    dataset['is_weekend'] = dataset.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset = pd.concat([dataset, weekday_dummies], axis=1)
    dataset['label'] = dataset.date.astype('str') + ':' + dataset.date_received.astype('str')
    dataset.label = dataset.label.apply(get_label)
    dataset.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1,
                  inplace=True)
    dataset = dataset.replace('null', np.nan)
    dataset.to_csv(TraingSetPath, index=None)
    print("Successfully generate Training Set!")
    print("Shape of Training Set is:")
    print(dataset.shape)

def GenTestingSet(FeaturePath,datasetIndex):
    TestingSetPath = os.path.join(FeaturePath,'dataset'+str(datasetIndex)+'.csv')
    if os.path.exists(TestingSetPath):
        print("%s has existed!" % TestingSetPath)
        return
    coupon = pd.read_csv(os.path.join(FeaturePath,"coupon"+str(datasetIndex)+"_feature.csv"))
    merchant = pd.read_csv(os.path.join(FeaturePath,"merchant"+str(datasetIndex)+"_feature.csv"))
    user = pd.read_csv(os.path.join(FeaturePath,"user"+str(datasetIndex)+"_feature.csv"))
    user_merchant = pd.read_csv(os.path.join(FeaturePath,"user_merchant"+str(datasetIndex)+".csv"))
    other_feature = pd.read_csv(os.path.join(FeaturePath,"other_feature"+str(datasetIndex)+".csv"))
    dataset = pd.merge(coupon, merchant, on='merchant_id', how='left')
    dataset = pd.merge(dataset, user, on='user_id', how='left')
    dataset = pd.merge(dataset, user_merchant, on=['user_id', 'merchant_id'], how='left')
    dataset = pd.merge(dataset, other_feature, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset.drop_duplicates(inplace=True)

    dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan, 0)
    dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan, 0)
    dataset.user_merchant_received = dataset.user_merchant_received.replace(np.nan, 0)
    dataset['is_weekend'] = dataset.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset = pd.concat([dataset, weekday_dummies], axis=1)
    dataset.drop(['merchant_id', 'day_of_week', 'coupon_count'], axis=1, inplace=True)
    dataset = dataset.replace('null', np.nan)

    dataset.to_csv(TestingSetPath, index=None)
    print("Successfully generate Testing Set!")
    print("Shape of Testing Set is:")
    print(dataset.shape)

if __name__ == '__main__':
    DataDir = os.path.join(os.path.dirname(os.getcwd()),'data')
    FeaturePath = os.path.join(DataDir, 'DataFeatures')
    GenTestingSet(FeaturePath,3)
    GenTraningSet(FeaturePath,2)
    GenTraningSet(FeaturePath,1)
