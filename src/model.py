import clean_datasets as clean
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
import prince
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

ad = clean.ad_cleaned()
mod = clean.mod_cleaned()
ad
mod.index = range(1,len(mod)+1)
mod
#standardising data
scaler = StandardScaler()
standardised_cols = scaler.fit_transform(mod.loc[:,'productivity':'accuracy'])
cols = ['productivity', 'utilisation_percentage','handling_time','accuracy']
df = pd.DataFrame(standardised_cols,columns = cols)
df.index = df.index + 1

df.shape
mod['productivity'] = df['productivity']
mod['utilisation_percentage'] = df['utilisation_percentage']
mod['handling_time'] = df['handling_time']
mod['accuracy'] = df['accuracy']
mod
mod.isnull().sum()


sk_pca = PCA(n_components=4,random_state=234)
sk_pca.fit(df)
sk_pca
dset2 = pd.DataFrame()
dset2['pca'] = range(1,5)
dset2['vari'] = pd.DataFrame(sk_pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
graph = sns.barplot(x='pca', y='vari', data=dset2)
for p in graph.patches:
    graph.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),
                   ha='center', va='bottom',
                   color= 'black')
plt.ylabel('Proportion', fontsize=18)
plt.xlabel('Principal Component', fontsize=18)
plt.show()




    

