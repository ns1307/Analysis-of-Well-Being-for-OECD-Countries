import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("skill_mismatch.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
mismatch_titles=["Field of Study Mismatch","Overqualification","Qualification Mismatch","Underqualification"]
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]
mismatch_queries=["FOS","OQ","QM","UQ"]

years=list(np.arange(2000,2013))
table = pd.DataFrame(index=countries, columns=["FOS","OQ","QM","UQ"])
no_data_countries=[]
new_countries=[]
new_colors=[]
def getFromFiltered(df, column,row):
    if df.loc[(df[column]==row)].empty:
        return 0
    else:
        return df.loc[(df[column]==row)].Value.to_numpy()[0]
for i in range(len(countries)):
    country=countries[i]
    countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==country)]#all data beloning to that country

    if countr_filtered_df.empty:#country has no data
        no_data_countries.append(country)
        colors.remove(colors[i])
        table = table.drop(country)
    else:
        FOS=getFromFiltered(countr_filtered_df,"MISMATCH","FOS")
        OQ=getFromFiltered(countr_filtered_df,"MISMATCH","OQ")
        QM=getFromFiltered(countr_filtered_df,"MISMATCH","QM")
        UQ=getFromFiltered(countr_filtered_df,"MISMATCH","UQ")
        table["FOS"][country]=FOS
        table["OQ"][country]=OQ
        table["QM"][country]=QM
        table["UQ"][country]=UQ
        new_countries.append(country)
      
countries=new_countries

def check_similarity(value1,value2):
    
    #similarity is 1 if they are equal, It can go to 0 as the values of numbers differ
    if value1==0 and value2==0:
        similarity=1
    else:
        similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
    return similarity,1




for i in range (len(mismatch_queries)):
    mismatch=mismatch_queries[i]
    similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    for country in countries:
        value1=table.loc[country][mismatch]
        for other_country in countries:
            value2=table.loc[other_country][mismatch]
            similarity, number_of_values = check_similarity(value1,value2)
            similarity_df[country][other_country]=similarity
            similarity_numbers[country][other_country]=number_of_values
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+mismatch_titles[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+mismatch_titles[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    num_cluster=6
    from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
    
    data_matrix = similarity_df
    model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
    
    cluster_table = pd.DataFrame(columns=["cluster","countries"])
    print("Clusters for "+mismatch_titles[i])
    print(model.labels_)
    cluster_array=model.labels_
    clusters=[]
    for a in range (num_cluster):
        cluster=a
        cluster_name="Cluster "+str(cluster)
        array=[]
        for ind in range (len(countries)):
            if cluster_array[ind]==cluster:
                array.append(countries[ind])
        if len(array)!=0:
               df2 = {"cluster":cluster_name,'countries': array}
               cluster_table = cluster_table.append(df2, ignore_index = True)
        clusters.append(array)
        print("Cluster ",cluster,": ",array)
    print("----------------------")

    fig = ff.create_table(cluster_table)

    fig.update_layout(
    autosize=True,

    )

    fig.write_image("Clusters "+mismatch_titles[i]+".png", scale=2)

for i in range (len(mismatch_titles)):
    country_names=countries.copy()
    mismatch=mismatch_queries[i]
    data=table[mismatch]
    j=0
    
    
    fig = plt.figure(figsize=(8,3))
    
    
    
    title=mismatch_titles[i]
    
    plt.bar(country_names, data, color =colors,
            width = 0.4)
     
    plt.xlabel("Countries")
    plt.ylabel("Percentage")
    plt.title(title)
    
    text="No data for: "+', '.join(no_data_countries)
    plt.figtext(0,0, text, ha="left", fontsize=14)
    plt.tight_layout()
    plt.savefig(title+".png", dpi=500)
    plt.show()