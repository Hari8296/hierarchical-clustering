# Name = Hari singh r
# batch id = DSWDMCOD 25082022 B

1
Business problems 
1.1 what is the Business odjectives 
Ans:- Maximize the profit
      minimize the cost of airlines to compare other airlines 
      
1.2 What are the constraints      
Ans:- provide offers on air tickicts to compare other airlines 


import pandas as pd 
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

df = pd.read_excel("D:/assignments of data science/07 hierarchical clustering/EastWestAirlines.xlsx",sheet_name= 1)


3
df
df.dtypes
df.sum()
df.isna().sum()
df.duplicated().sum()

df1=df.drop(["ID#","Award?"],axis=1)
df1

df1.mean()
df1.median()
df1.mode()
list(df1.mode())
df1.skew()
df1.kurt()
df1.var()

for i in df1.columns:
    plt.hist(df1[i])
    plt.xlabel(i)
    plt.show()
    
for i in df1.columns:
    for j in df1.columns:
        if(i!=j):
            plt.scatter(df1[i],df1[j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()
    
from sklearn.preprocessing import StandardScaler 
stn_df=StandardScaler().fit_transform(df1)    
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

df1['clust']=cluster_labels
   
df1["clust"]=clust(stn_df,5)    
df1["clust"].value_counts()  

df2=df1
df3=df1 
df4=df1   

df2["clust"]=clust(stn_df,7)
df2["clust"].value_counts()

df3["clust"]=clust(stn_df,3)
df3["clust"].value_counts()    

6

increase in credit points when travel is frequenty using the our flight 


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    