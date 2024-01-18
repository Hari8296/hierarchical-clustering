Name:- Hari singh r
batch Id :- DSWDMCOD 25082022 B

1
Business problems 
maximize the security with camers
minimize the crime rate in city

2
buissness constrain
giving awerness of crime 

import pandas as pd
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

df=pd.read_csv("D:/assignments of data science/07 hierarchical clustering/crime_data.csv")
df

df.dtypes
df.duplicated().sum()
df.isna().sum()

df.rename(columns={'Unnamed: 0':'state'},inplace=True)

df.mean()
df.median()
df.mode()
list(df.mode())
df.skew()
df.kurt()
df.var()

plt.figure(figsize=(20,5))
df.groupby('state')['Murder'].sum().plot(kind='bar')


plt.figure(figsize=(20,5))
df.groupby('state')['Assault'].sum().plot(kind='bar')


plt.figure(figsize=(20,5))
df.groupby('state')['Rape'].sum().plot(kind='bar')


plt.figure(figsize=(20,5))
df.groupby('state')['UrbanPop'].sum().plot(kind='bar')

for i in ['Murder','Assault','Rape']:
            plt.scatter(df["UrbanPop"],df[i])
            plt.xlabel('UrbanPop')
            plt.ylabel(i)
            plt.show()

from sklearn.preprocessing import StandardScaler 
stn_df=StandardScaler().fit_transform(df.iloc[:,1:])    
stn_df    


from scipy.cluster.hierarchy import linkage,dendrogram    
z=linkage(stn_df,method="ward",metric="euclidean") 

plt.figure(figsize=(15,8));plt.title("dendrogram");plt.xlabel("index");plt.ylabel("distance")
dendrogram(z,
           leaf_rotation=0,
           leaf_font_size=10
           )
plt.show()

def clust(df,n_clust):
    from sklearn.cluster import AgglomerativeClustering
    h_complete=AgglomerativeClustering(n_clusters=n_clust,linkage="ward",affinity="euclidean").fit(df)
    var=h_complete.labels_
    cluster_labels=pd.Series(var) 
    return cluster_labels

df["clust"]=clust(stn_df,4)    
df["clust"].value_counts()  

df1=df

df1["clust"]=clust(stn_df,3)    
df1["clust"].value_counts()  

df.groupby("clust").mean()


in the citizen of clust 0:-it has high crime rate in everthing which less safer then compare to others 
in the citizen of clust 1:-it has less crime rate when compare to clust 0 citizen which make to safe to stay 
in the citizen pf clust 2:-it has less crime rate when compare to clust 0 and 1 which make more safer to stay in clust 2 city for better life 
























