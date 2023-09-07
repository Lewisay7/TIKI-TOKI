import numpy as np
import pandas as pd

url= "TIKI-TOKI/data/ad_ranking_raw_data.xlsx"
ad_data = pd.read_excel(url,sheet_name=0,header=1)

ad_data.head()
ad_data.ad_id.is_unique
pd.unique(ad_data["queue_market"])



