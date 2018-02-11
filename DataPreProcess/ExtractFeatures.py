import pandas as pd
import numpy as np
from datetime import date
import os
import re

"""
目前做的特征工程：
    1.merchant related: 
          sales_use_coupon. total_coupon
          transfer_rate = sales_use_coupon/total_coupon.
          merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon 
          total_sales.  coupon_rate = sales_use_coupon/total_sales.  

    2.coupon related: 
          discount_rate. discount_man. discount_jian. is_man_jian
          day_of_week,day_of_month. (date_received)

    3.user related: 
          distance. 
          user_avg_distance, user_min_distance,user_max_distance. 
          buy_use_coupon. buy_total. coupon_received.
          buy_use_coupon/coupon_received. 
          avg_diff_date_datereceived. min_diff_date_datereceived. max_diff_date_datereceived.  
          count_merchant.  

    4.user_merchant:
          times_user_buy_merchant_before.


    5. other feature:
          this_month_user_receive_all_coupon_count
          this_month_user_receive_same_coupon_count
          this_month_user_receive_same_coupon_lastone
          this_month_user_receive_same_coupon_firstone
          this_day_user_receive_all_coupon_count
          this_day_user_receive_same_coupon_count
          day_gap_before, day_gap_after  (receive the same coupon)
"""


def splitData(DataDir):
    '''
    将数据用滑动窗口划分为训练集和数据集（按照时间），其中训练集为三个月，
    验证集和测试集一个月,然后将划分好的数据集保存到csv文件
    :param DataDir: 数据存放的目录
    :return: None
    '''
    if not os.path.exists(DataDir):
        print("No %s exists!" % DataDir)
        return
    OriginDataPath = os.path.join(DataDir, 'OriginData')
    SplitDataPath = os.path.join(DataDir, 'SplitData')
    off_train = pd.read_csv(os.path.join(OriginDataPath, 'ccf_offline_stage1_train.csv'))
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    # 2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
    off_test = pd.read_csv(os.path.join(OriginDataPath, 'ccf_offline_stage1_test_revised.csv'))
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    # 11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
    on_train = pd.read_csv(os.path.join(OriginDataPath, 'ccf_online_stage1_train.csv'))
    on_train.columns = ['user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate', 'date_received', 'date']

    dataset3 = off_test
    dataset3.to_csv(os.path.join(SplitDataPath, 'dataset3.csv'), index=None)
    feature3 = off_train[((off_train.date >= '20160315') & (off_train.date <= '20160630')) | (
        (off_train.date == 'null') & (off_train.date_received >= '20160315') & (off_train.date_received <= '20160630'))]
    feature3.to_csv(os.path.join(SplitDataPath, 'feature3.csv'), index=None)
    dataset2 = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
    dataset2.to_csv(os.path.join(SplitDataPath, 'dataset2.csv'), index=None)
    feature2 = off_train[(off_train.date >= '20160201') & (off_train.date <= '20160514') | (
        (off_train.date == 'null') & (off_train.date_received >= '20160201') & (off_train.date_received <= '20160514'))]
    feature2.to_csv(os.path.join(SplitDataPath, 'feature2.csv'), index=None)
    dataset1 = off_train[(off_train.date_received >= '20160414') & (off_train.date_received <= '20160514')]
    dataset1.to_csv(os.path.join(SplitDataPath, 'dataset1.csv'), index=None)
    feature1 = off_train[(off_train.date >= '20160101') & (off_train.date <= '20160413') | (
        (off_train.date == 'null') & (off_train.date_received >= '20160101') & (off_train.date_received <= '20160413'))]
    feature1.to_csv(os.path.join(SplitDataPath, 'feature1.csv'), index=None)


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  # those only receive once


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (
            date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]),
                                                                                                   int(d[4:6]),
                                                                                                   int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]),
                                                                       int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def ExtractOtherFeature(dataset, FeaturePath):
    """
    other feature:
          this_month_user_receive_all_coupon_count
          this_month_user_receive_same_coupon_count
          this_month_user_receive_same_coupon_lastone
          this_month_user_receive_same_coupon_firstone
          this_day_user_receive_all_coupon_count
          this_day_user_receive_same_coupon_count
          day_gap_before, day_gap_after  (receive the same coupon)
    """
    if os.path.exists(FeaturePath):
        print("%s has existed!" % FeaturePath)
        return
    t = dataset.loc[:, ['user_id']]
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset.loc[:, ['user_id', 'coupon_id']]
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

    t2 = dataset.loc[:, ['user_id', 'coupon_id', 'date_received']]
    t2.date_received = t2.date_received.astype('str')
    t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset.loc[:, ['user_id', 'coupon_id', 'date_received']]
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    t4 = dataset.loc[:, ['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    t5 = dataset.loc[:, ['user_id', 'coupon_id', 'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    t6 = dataset.loc[:, ['user_id', 'coupon_id', 'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    """
    other feature:
          t(['user_id']):this_month_user_receive_all_coupon_count
          t1(['user_id', 'coupon_id']):this_month_user_receive_same_coupon_count
          t2,t3(['user_id', 'date_received']):
                this_month_user_receive_same_coupon_lastone#t3已经把t2合并
          t4,t5(['user_id', 'coupon_id', 'date_received']):
                this_month_user_receive_same_coupon_firstone
                this_day_user_receive_all_coupon_count
                this_day_user_receive_same_coupon_count
          t7,t6(['user_id', 'coupon_id', 'date_received']):
                day_gap_before, day_gap_after  (receive the same coupon)
    """
    other_feature = pd.merge(t1, t, on='user_id')
    other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
    other_feature = pd.merge(other_feature, t4, on=['user_id', 'date_received'])
    other_feature = pd.merge(other_feature, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature = pd.merge(other_feature, t7, on=['user_id', 'coupon_id', 'date_received'])
    other_feature.to_csv(FeaturePath, index=None)
    print("Successfully extract Other Features!")
    print("Shape of Other Features is:")
    print(other_feature.shape)


def calc_discount_rate(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return float(s[0])
    else:
        return 1.0 - float(s[1]) / float(s[0])


def get_discount_man(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[0])


def get_discount_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 'null'
    else:
        return int(s[1])


def is_man_jian(s):
    s = str(s)
    s = s.split(':')
    if len(s) == 1:
        return 0
    else:
        return 1


def ExtractCouponRelatedFeature(dataset, FeaturePath):
    """
    coupon related: 
          discount_rate. discount_man. discount_jian. is_man_jian
          day_of_week,day_of_month. (date_received)
    """
    if os.path.exists(FeaturePath):
        print("%s has existed!" % FeaturePath)
        return
    dataset['day_of_week'] = dataset.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    dataset['day_of_month'] = dataset.date_received.astype('str').apply(lambda x: int(x[6:8]))
    dataset['days_distance'] = dataset.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days)
    dataset['discount_man'] = dataset.discount_rate.apply(get_discount_man)
    dataset['discount_jian'] = dataset.discount_rate.apply(get_discount_jian)
    dataset['is_man_jian'] = dataset.discount_rate.apply(is_man_jian)
    dataset['discount_rate'] = dataset.discount_rate.apply(calc_discount_rate)
    d = dataset.loc[:, ['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, d, on='coupon_id', how='left')
    dataset.to_csv(FeaturePath, index=None)
    print("Successfully extract Coupon Related Feature!")
    print("Shape of Coupon Related Feature is:")
    print(dataset.shape)


def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (
    date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days


def ExtractUserRelatedFeature(feature, FeaturePath):
    """
    user related: 
          count_merchant. 
          user_avg_distance, user_min_distance,user_max_distance,,user_median_distance. 
          buy_use_coupon. buy_total. coupon_received.
          buy_use_coupon/coupon_received. 
          buy_use_coupon/buy_total
          user_date_datereceived_gap
    """
    if os.path.exists(FeaturePath):
        print("%s has existed!" % FeaturePath)
        return
    user = feature.loc[:, ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

    t = user.loc[:, ['user_id']]
    t.drop_duplicates(inplace=True)

    t1 = user[user.date != 'null'].loc[:, ['user_id', 'merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id').agg('sum').reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    t2 = user[(user.date != 'null') & (user.coupon_id != 'null')].loc[:, ['user_id', 'distance']]
    t2.replace('null', -1, inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    t4 = t2.groupby('user_id').agg('max').reset_index()
    t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    t5 = t2.groupby('user_id').agg('mean').reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    t6 = t2.groupby('user_id').agg('median').reset_index()
    t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')].loc[:, ['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id').agg('sum').reset_index()

    t8 = user[user.date != 'null'].loc[:, ['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id').agg('sum').reset_index()

    t9 = user[user.coupon_id != 'null'].loc[:, ['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id').agg('sum').reset_index()

    t10 = user[(user.date_received != 'null') & (user.date != 'null')].loc[:, ['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    t11 = t10.groupby('user_id').agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    t12 = t10.groupby('user_id').agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    t13 = t10.groupby('user_id').agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    user_feature = pd.merge(t, t1, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t3, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t4, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t5, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t6, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t7, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t8, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t9, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t11, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t12, on='user_id', how='left')
    user_feature = pd.merge(user_feature, t13, on='user_id', how='left')
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.buy_total.astype('float')
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    user_feature.to_csv(FeaturePath, index=None)
    print("Successfully extract User Related Feature!")
    print("Shape of User Related Feature is:")
    print(user_feature.shape)


def ExtractUserMerchantRelatedFeature(feature, FeaturePath):
    """
    user_merchant:
          times_user_buy_merchant_before. 
    """
    if os.path.exists(FeaturePath):
        print("%s has existed!" % FeaturePath)
        return
    all_user_merchant = feature.loc[:, ['user_id', 'merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    t = feature.loc[:, ['user_id', 'merchant_id', 'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    t1 = feature.loc[:, ['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    t2 = feature.loc[:, ['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    t3 = feature.loc[:, ['user_id', 'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    t4 = feature.loc[:, ['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t1, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t2, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t3, on=['user_id', 'merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t4, on=['user_id', 'merchant_id'], how='left')
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype(
        'float') / user_merchant.user_merchant_any.astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant.to_csv(FeaturePath, index=None)
    print("Successfully extract User-Merchant Related Feature!")
    print("Shape of User-Merchant Related Feature is:")
    print(user_merchant.shape)


def ExtractMerchantRelatedFeature(feature, FeaturePath):
    """
    merchant related: 
          total_sales. sales_use_coupon.  total_coupon
          coupon_rate = sales_use_coupon/total_sales.  
          transfer_rate = sales_use_coupon/total_coupon. 
          merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon
    """
    if os.path.exists(FeaturePath):
        print("%s has existed!" % FeaturePath)
        return
    merchant = feature.loc[:, ['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]

    t = merchant.loc[:, ['merchant_id']]
    t.drop_duplicates(inplace=True)

    t1 = merchant[merchant.date != 'null'].loc[:, ['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id').agg('sum').reset_index()

    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')].loc[:, ['merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id').agg('sum').reset_index()

    t3 = merchant[merchant.coupon_id != 'null'].loc[:, ['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id').agg('sum').reset_index()

    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')].loc[:, ['merchant_id', 'distance']]
    t4.replace('null', -1, inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1, np.nan, inplace=True)
    t5 = t4.groupby('merchant_id').agg('min').reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    t6 = t4.groupby('merchant_id').agg('max').reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)

    t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)

    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    merchant_feature = pd.merge(t, t1, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t2, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t3, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t5, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t6, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t7, on='merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t8, on='merchant_id', how='left')
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_coupon
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_sales
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    merchant_feature.to_csv(FeaturePath, index=None)
    print("Successfully extract Merchant Related Feature!")
    print("Shape of Merchant Related Feature is:")
    print(merchant_feature.shape)


if __name__ == '__main__':
    DataDir = os.path.join(os.path.dirname(os.getcwd()),'data')
    SplitDataPath = os.path.join(DataDir, 'SplitData')
    OriginDataPath = os.path.join(DataDir, 'OriginData')
    if len(os.listdir(SplitDataPath)) == 0:
        splitData(DataDir)
        print("Successfully Split the Original data!")
    else:
        print("The original Data have been split!")

    for csvFileName in os.listdir(SplitDataPath):
        FeaturePath = os.path.join(DataDir, 'DataFeatures')
        datasetIndex = int(re.findall('.*(\d+)\.csv', csvFileName)[0])
        if csvFileName.find("dataset") != -1:
            dataset = pd.read_csv(os.path.join(SplitDataPath, csvFileName))
            ExtractOtherFeature(dataset, os.path.join(FeaturePath, 'other_feature' + str(datasetIndex) + '.csv'))
            ExtractCouponRelatedFeature(dataset,
                                        os.path.join(FeaturePath, 'coupon' + str(datasetIndex) + '_feature.csv'))
        elif csvFileName.find("feature") != -1:
            feature = pd.read_csv(os.path.join(SplitDataPath, csvFileName))
            ExtractUserRelatedFeature(feature, os.path.join(FeaturePath, 'user' + str(datasetIndex) + '_feature.csv'))
            ExtractUserMerchantRelatedFeature(feature,
                                              os.path.join(FeaturePath, 'user_merchant' + str(datasetIndex) + '.csv'))
            ExtractMerchantRelatedFeature(feature,
                                          os.path.join(FeaturePath, 'merchant' + str(datasetIndex) + '_feature.csv'))
