Name :- Hari singh r
batch Id :- batch id = DSWDMCOD 25082022 B

buissness objectives
Maximize-revenue
Minimize-customer churn

buissness constrain
to decrease customer churn will need efficent services 

import pandas as pd
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

df=pd.read_excel("D:/assignments of data science/07 hierarchical clustering/Telco_customer_churn.xlsx")
df

df.dtypes
df.isna().sum()
df.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
en_df= df.iloc[:, [3,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23]]
nen_df= df.iloc[:, [0,1,2,4,5,8,12,24,25,26,27,28,29]]

for i in en_df.columns:
    en_df[i]=labelencoder.fit_transform(en_df[i])
    
 new_df=pd.concat([en_df,nen_df],axis=1)   

df.mean()
df.median()
df.mode()
list(df.mode())
df.skew()
df.kurt()
df.var()

for i in df.columns:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()

plt.scatter(df["Tenure in Months"], df["Total Revenue"])
plt.xlabel('Tenure in Months')
plt.ylabel('Total Revenue')

from sklearn.preprocessing import StandardScaler 
stn_df=StandardScaler().fit_transform(new_df.drop(['Customer ID','Count','Quarter'],axis=1))    
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

  
df["clust"]=clust(stn_df,3)    
df["clust"].value_counts()  

df1=df

df1["clust"]=clust(stn_df,4)    
df1["clust"].value_counts()  

new_df["clust"]=clust(stn_df,3)
z=new_df.groupby("clust").mean()

Clust 0 = the customers that are frequent users The revenue earned through these is also the best. Hence, these are the customers that are least likely to churn.
Clust 2 = the customers may or may not churn that frequently.
clust 1 = the customers that are least frequent.these are the ones that churn chances is more



























