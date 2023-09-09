import numpy as np
import pandas as pd
from datetime import date, datetime

import os 
try : 
    os.chdir("src")
except FileNotFoundError: 
    pass

'''Cleaning advertisement data'''
def ad_cleaned():
    #data read in 
    url= "../data/ad_ranking_raw_data.xlsx"
    ad = pd.read_excel(url,sheet_name=0,header=1)
    ad = pd.DataFrame(ad)
    ad.drop(["product_line", "task_type_en"], axis='columns', inplace=True)
    
    #clean up columns
    ad["queue_market"]=np.where(ad["queue_market"]=="US&CA","USCA",ad["queue_market"])
    ad["p_date"] = pd.to_datetime(ad["p_date"],format ='%Y%m%d')
    ad.drop_duplicates( subset = "ad_id")
    ad.round({'ad_revenue':3,'avg_ad_revenue':3,'baseline_st':3})
    ad["punish_num"] = ad["punish_num"].replace(np.nan, 0)

    null_qm = np.where(ad["queue_market"].isnull())
    for row in null_qm[0]:  # Use null_qm[0] to get the row indices
        ad.loc[row, "queue_market"] = ad.loc[row, "delivery_country"]

    null_rev = np.where(ad["ad_revenue"].isnull())
    for row in null_rev[0]:  # Use null_rev[0] to get the row indices
        ad.loc[row, "ad_revenue"] = ad.loc[row, "avg_ad_revenue"]

    ad.dropna(inplace=True)
    ad.index = range(1,len(ad)+1)
    
    duration_since_start_time = []  
    today = date.today().strftime("%Y/%m/%d")
    for start_time in ad["start_time"]:
        start_time_str = start_time.strftime("%Y/%m/%d")
        d1 = datetime.strptime(today, "%Y/%m/%d")
        d2 = datetime.strptime(start_time_str, "%Y/%m/%d")
        
        duration = (d1 - d2).days
        duration_since_start_time.append(duration)

    ad["duration_since_start_time"] = duration_since_start_time
    return ad

ad_cleaned()


'''Cleaning moderator data'''
def replace_0(predictors,data):
    for i in predictors:
        df = data[i]
        data[i] = df.replace('                 -  ',0)
    return data

def replace_nan(predictors, data):
    for i in predictors:
        df = data[i]
        data[i] = df.replace(0, np.nan)
    return data

def mod_cleaned():
    #read in data
    url= "../data/ad_ranking_raw_data.xlsx"
    col_names = ['mod_id', 'market','productivity', 'utilisation_percentage','handling_time','accuracy']
    moderator_data = pd.read_excel(url,sheet_name=1,header=None,skiprows=1,names =['mod_id', 'market','productivity', 'utilisation_percentage','handling_time','accuracy'])
    #replacing obs with '-' to 0
    moderator_data = replace_0(predictors = col_names,data = moderator_data)
    #replacing 0 to nan
    moderator_data = replace_nan(predictors = col_names,data = moderator_data)
    #removing obs with nan
    moderator_data.dropna(inplace=True)
    #dropping duplicated mod_id obs if there is any
    moderator_data.drop_duplicates(subset=['mod_id'])
    #rounding off utilisation,productivity to 2 dp and accuracy to 3 dp
    moderator_data = moderator_data.round({'utilisation_percentage':2, 'productivity':2, 'accuracy':3})
    moderator_data.index = range(1,len(moderator_data)+1)
    return moderator_data

mod_cleaned()








