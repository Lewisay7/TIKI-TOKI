import clean_datasets as clean
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plt
from sklearn.preprocessing import StandardScaler

ad = clean.ad_cleaned()
mod = clean.mod_cleaned()
ad
mod

#standardising data
scaler = StandardScaler()
standardised_cols = scaler.fit_transform(mod.loc[:,'productivity':'accuracy'])
cols = ['productivity', 'utilisation_percentage','handling_time','accuracy']
df = pd.DataFrame(standardised_cols,columns = cols)
df.index = df.index + 1
df.shape
mod['productivity'] = df['productivity']
mod['productivity']
mod.isnull().sum()





    

