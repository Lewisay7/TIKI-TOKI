import clean_datasets as clean
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

ad = clean.ad_cleaned()

'''ADVERTISEMENT MODEL'''
#standardising data
scaler = StandardScaler()
cols = ['punish_num','ad_revenue','baseline_st','duration_since_start_time']
#converting duration from time to int 
ad['duration_since_start_time'] = ad['duration_since_start_time'].dt.days
#preparing data to analyse correlation and PCA

ad_1 = ad.drop(['latest_punish_begin_date','avg_ad_revenue','start_time'], axis = 1)

#standardising data
standardised_cols = scaler.fit_transform(ad_1.loc[:,'punish_num':'duration_since_start_time'])
standardised_cols
df = pd.DataFrame(standardised_cols,columns = cols)
df.index = df.index + 1

ad_1['punish_num'] = df['punish_num']
ad_1['ad_revenue'] = df['ad_revenue']
ad_1['baseline_st'] = df['baseline_st']
ad_1['duration_since_start_time'] = df['duration_since_start_time']
ad_1
ad_1.isnull().sum()

#visualising relationships between features
sns.set(style="whitegrid")
sns.pairplot(ad_1.loc[:,'punish_num':'duration_since_start_time'], kind='reg', diag_kind='kde')
plt.show()

#Correlation
plt.figure(figsize=(8,6))
sns.set(style="whitegrid")
sns.heatmap(ad_1.loc[:,'punish_num':'duration_since_start_time'].corr(method='pearson'), vmin=-.1, vmax=1,  annot=True, cmap='RdYlGn')
plt.show()

#variance explained by principal components
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

#transforming the data using pca
ad2 = sk_pca.transform(df)

#assigning the scores using weights of the proportion of explained ratio by the principal components
scores = (ad2 * sk_pca.explained_variance_ratio_).sum(axis=1)

#scaling the scores into the range of 0 and 1 
scaler = MinMaxScaler()
scores = scores.reshape(-1, 1)
scores = scaler.fit_transform(scores)
scores

#Assigning scores to the original advertisement datasets
ad["scores"] = scores
ad
#sorting the ads based on scores
ad = ad.sort_values(by='scores', ascending= False)
ad












    

