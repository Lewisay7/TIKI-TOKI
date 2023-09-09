import numpy as np
import pandas as pd
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
    for row in null_qm:
        ad["queue_market"][row] = ad["delivery_country"][row]

    null_rev = np.where(ad["ad_revenue"].isnull())
    for row in null_rev:
        ad["ad_revenue"][row] = ad["avg_ad_revenue"][row]

    ad.dropna(inplace=True)

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
    return moderator_data

mod_cleaned()








