    Name = Hari singh r
batch id = DSWDMCOD 25082022 B

import pandas as pd
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

df=pd.read_csv("D:/assignments of data science/07 hierarchical clustering/AutoInsurance.csv")
df

1

#Business objectives
maximize the profit
minimize the claims 

#buissness constrain

better competetion on offer given by other companies

df.dtypes 
df.duplicated().sum()
df.isna().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
en_df= df.iloc[:, [3,4,5,7,8,10,11,17,18,19,20,22,23]]
nen_df= df.iloc[:, [0,1,2,6,9,12,13,14,15,16,21]]

for i in en_df.columns:
    en_df[i]=labelencoder.fit_transform(en_df[i])


new_df=pd.concat([nen_df,en_df],axis=1)

df.mean()
df.median()
list(df.mode())
df.skew()
df.kurt()
df.var()

for i in df.columns:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()
    
    
plt.scatter(df['Customer Lifetime Value'],df['Total Claim Amount'])
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Total Claim Amount')   
    
from sklearn.preprocessing import StandardScaler
stn_df=StandardScaler().fit_transform(new_df.drop(['Customer','State','Effective To Date'],axis=1))
stn_df    

from scipy.cluster.hierarchy import linkage,dendrogram     
z=linkage(stn_df,method="ward",metric='euclidean')    
plb.figure(figsize=(15,8))
plb.title("Dendrogram")
plb.xlabel("INDEX")
plb.ylabel("DISTANCE")
dendrogram(z,
           leaf_rotation=0,
           leaf_font_size=10     
           )
plb.show()

def clust(df,n_clust):
    from sklearn.cluster import AgglomerativeClustering
    h_complete=AgglomerativeClustering(n_clusters=n_clust,linkage="ward",affinity="euclidean").fit(df)
    var=h_complete.labels_
    cluster_labels=pd.Series(var) 
    return cluster_labels

df["clust"]=clust(stn_df,5)    
df["clust"].value_counts()  

df1=df
df2=df 

df1["clust"]=clust(stn_df,4)
df1["clust"].value_counts()

new_df["clust"]=clust(stn_df,3)
z=new_df.groupby("clust").mean()

cluster 0 = has chance of most income.
cluster 1 = has less income chance than cluster 0.
cluster 2 = has the maximum claim amount.

the insurance company has to focus on "cluster o" because there has more incomeand there is less cliams which makes company profit 



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
