import pandas as pd
import numpy as np
import cloudpickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict
import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

pd.set_option('display.max_columns',None)

from warnings import filterwarnings
filterwarnings('ignore')

def transform_categories(df):
    series = df['categories.name']
    series = [i.lower() for i in series]

    from collections import defaultdict
    new_features = defaultdict(list)


    for i in series:

        new_features['fun'].append(1) if 'fun' in i else new_features['fun'].append(0)
        new_features['mile'].append(1) if 'mile' in i else new_features['mile'].append(0)
        new_features['run'].append(1) if 'run' in i else new_features['run'].append(0)
        new_features['walk'].append(1) if 'walk' in i else new_features['walk'].append(0)
        new_features['wheelchair'].append(1) if 'wheelchair' in i else new_features['wheelchair'].append(0)
        new_features['commitment'].append(1) if 'commitment' in i else new_features['commitment'].append(0)
        new_features['bike'].append(1) if 'bike' in i else new_features['bike'].append(0)
        new_features['half marathon'].append(1) if 'half marathon' in i else new_features['half marathon'].append(0)
        new_features['marathon'].append(1) if 'marathon' in i else new_features['marathon'].append(0)
        new_features['fire fighter'].append(1) if 'fire fighter' in i else new_features['fire fighter'].append(0)
        new_features['quarter marathon'].append(1) if 'quarter marathon' in i else new_features['quarter marathon'].append(0)
        new_features['mini-marathon'].append(1) if 'mini-marathon' in i else new_features['mini-marathon'].append(0)
        new_features['run swim run'].append(1) if 'run swim run' in i else new_features['run swim run'].append(0)
        new_features['10k'].append(1) if '10k' in i else new_features['10k'].append(0)
        
        
        

        if 'triathlon' in i:
            new_features['tri'].append(1)
        else:
            new_features['tri'].append(0)

        if 'duathlon' in i or 'run swim run' in i:
            new_features['dua'].append(1)
        else:
            new_features['dua'].append(0)

    new_features = pd.DataFrame(new_features)
    a = new_features.sum(axis=1)
    a = np.where(a.values==0, 1, 0)
    new_features['run'] = np.where(new_features['run']+a>0, 1, 0)
    new_features['all'] =1
    return new_features


def transform_distance(df):
    dist = df['category.registered.distance.quantity'].values
    dist = np.where(df['category.registered.distance.unit']=='mi',dist*1.609,dist)
    return pd.Series(dist,name='distance')


def transform_sex(df):
    new_sex = df['sex'].replace('F','Female').replace('M','Male').replace('NOT SPECIFIED','Unspecified')
    sex_columns = {
        'Female':[0] * len(new_sex),
        'Male':[0] * len(new_sex),
        'Unspecified':[0] * len(new_sex)
    }

    for index,i in enumerate(new_sex):
        if i in sex_columns.keys():
            sex_columns[i][index] = 1
    return pd.DataFrame(sex_columns)

def transform_travel_level(df):
    location = defaultdict(list)
    for i in df['hometown']:
        i=i.split(', ')
        try:
            location['home_city'].append(i[0].strip())
        except:
            location['home_city'].append(-1)
        try:
            location['home_state'].append(i[1].strip())
        except:
            location['home_state'].append(-1)

        if len(location['home_city']) > len(location['home_state']):
            break

    for i in df['location.city']:
        location['event_city'].append(i.strip())

    for i in df['location.state']:
        location['event_state'].append(i.strip())

    location = pd.DataFrame(location)
    travel_level = []
    for index,row in location.iterrows():
        if row['home_state'] !=row['event_state']:
            travel_level.append(3)
        elif row['home_city'] !=row['event_city']:
            travel_level.append(2)
        elif row['home_city'] == row['event_city']:
            travel_level.append(1)
        else:
            print(row)
            break
    return pd.Series(travel_level, name='travel_level')

def transform_participants_info(df):
    partici_info = df[['counts.participants.expected','counts.participants.registered']]
    partici_info['counts.participants.expected'] = partici_info['counts.participants.expected'].replace('150-300','300').astype('float')
    partici_info['counts.participants.registered'] = partici_info['counts.participants.registered'].astype('float')
    partici_info['partici_quit_count'] = partici_info['counts.participants.registered'] - partici_info['counts.participants.expected']
    partici_info['partici_quit_frac'] = (partici_info['counts.participants.registered'] - partici_info['counts.participants.expected'])/partici_info['counts.participants.registered']
    return partici_info

def transform_founding(df):
    a = pd.Series(np.where(df['fundraising.goal'].isnull(), 0, 1), name='fundraising.goal')
    return a

def transform_time(df):
    a = pd.to_datetime(df['category.registered.date'])
    month = pd.Series(a.dt.month, name='month')
    doy = pd.Series(a.dt.day_of_year, name='doy')
    week_of_day = pd.Series(a.dt.day_of_week, name='week_of_day')
    year = pd.Series(a.dt.year, name='year')

    b = pd.to_datetime(df['checkin_time'])
    checkin_time = pd.Series(b.dt.hour * 3600 + b.dt.minute * 60 + b.dt.second, name='checkin_time')

    return pd.concat([month,doy,week_of_day,year,checkin_time], axis=1)


def honest_transform_event(df,event_name):
    from collections import defaultdict
    honest_event = defaultdict(list)
    event_cat_comb = [str(a).strip()+'_'+str(b).strip() for a,b in zip(df['lineage.event_series.id'].values, df['categories.name'].values)]
    event_cat_comb = np.array(event_cat_comb)
    for i in event_name:
        honest_event[i] = np.where(event_cat_comb==i, 1, 0)
    
    # for a,b,c in zip(df['lineage.event_series.id'].values, df['categories.name'].values, df['event_cat_comb'].values):
    #     print(a,b,c,str(a),str(b),str(a).strip(),str(b).strip(),str(a).strip()+'_'+str(b).strip())
    # print([i for i in df['event_cat_comb'].values if '580b7caa-fc8c-423e-bcc3-14045206f5f9' in str(i)])
    return pd.DataFrame(honest_event)

def get_categories_combination(df):
    return pd.Series([str(a).strip()+'_'+str(b).strip() for a,b in zip(df['lineage.event_series.id'].values, df['categories.name'].values)], name='event_cat_comb')

def get_personal_profile(df):
    df['sex'] = df['sex'].replace('F','Female').replace('M','Male').replace('NOT SPECIFIED','Unspecified')
    a = df.groupby(['firstname','lastname','sex','hometown','lineage.event_series.id','categories.name']).count()['lineage.event_series.name']
    experience = a.reset_index(drop=False).rename(columns={'lineage.event_series.name':'experience'})
    df['result.duration.chip.seconds'] = [i.seconds for i in pd.to_timedelta(df['result.duration.chip'])]
    a = df.groupby(['firstname','lastname','sex','hometown','lineage.event_series.id','categories.name']).mean()['result.duration.chip.seconds']
    mean_speed = a.reset_index(drop=False).rename(columns={'result.duration.chip.seconds':'mean_historical_speed'})
    res = pd.merge(experience, mean_speed, on=['firstname','lastname','sex','hometown','lineage.event_series.id','categories.name'],how='outer').fillna(-1)
    return res

def transform_profile(df,profile):
    df['sex'] = df['sex'].replace('F','Female').replace('M','Male').replace('NOT SPECIFIED','Unspecified')
    a = pd.merge(df,profile,on=['firstname','lastname','sex','hometown','lineage.event_series.id','categories.name'],how='left').fillna(-1)
    return a[['experience','mean_historical_speed']]