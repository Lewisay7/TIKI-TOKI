import numpy as np
import pandas as pd

def ad_cleaned():
    #data read in 
    url= "TIKI-TOKI/data/ad_ranking_raw_data.xlsx"
    ad = pd.read_excel(url,sheet_name=0,header=1)

    #clean up columns
    ad["queue_market"]=np.where(ad["queue_market"]=="US&CA","USCA",ad["queue_market"])
    ad["p_date"] = pd.to_datetime(ad["p_date"],format ='%Y%m%d')
    ad = ad.drop_duplicates( subset = "ad_id")
    ad = ad.round({'ad_revenue':3,'avg_ad_revenue':3,'baseline_st':3})
    return ad

ad_cleaned()