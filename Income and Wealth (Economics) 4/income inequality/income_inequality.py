import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.figure_factory as ff


dataframe= pd.read_csv("income_inequality.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]



currencies=[]
notes=["Gini coefficient: It ranges between 0 in the case of perfect equality and 1 in the case of perfect inequality.",
       "S80/S20: The ratio of the average income of the 20% richest to the 20% poorest.",
       "P90/P10: The ratio of the upper bound value of the ninth decile (i.e. the 10% of people with highest income) to that of the first decile",
       "P90/P50: The upper bound value of the ninth decile to the median income",
       "P50/P10 of median income to the upper bound value of the first decile.",
       "The Palma ratio is the share of all income received by the 10% people with highest disposable income divided by the share of all income received by the 40% people with the lowest disposable income."]

queries=["GINI","S80S20","P90P10","P90P50","P50P10","PALMA"]
titles=["Gini Coefficient","S80/S20 Ratio","P90/P10 Ratio","P90/P50 Ratio","P50/P10 Ratio","Palma Ratio"]
years=list(np.arange(2015,2021))
table = pd.DataFrame(index=countries, columns=queries)
def getFromDF(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df


all_df=[]
for i in range(len(queries)):
    new_df=getFromDF(dataframe,"SUBJECT",queries[i])
    all_df.append(new_df)


for i in range(len(countries)):
    country=countries[i]
    country_df_list=[]
    for x in range(len(queries)):
        query_country_df=getFromDF(all_df[x],"LOCATION",country)
        query_values=[]
        for year in years:
            query_country_year_df=query_country_df.loc[(query_country_df['TIME']==year)]
            if not query_country_year_df.empty:
                query_values.append(query_country_year_df.Value.to_numpy()[0])
        
        gini_value=sum(query_values)/len(query_values)
        table[queries[x]][country]=gini_value
       
        

def check_similarity(value1,value2):
    similarity=0
    if value1==value1 and value2==value2:#if values are not empty
        similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
    return similarity,1
for x in range(len(queries)):
    similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    for country in countries:
        value1=table.loc[country][queries[x]]
        for other_country in countries:
            value2=table.loc[other_country][queries[x]]
            similarity, number_of_values = check_similarity(value1,value2)
            similarity_df[country][other_country]=similarity
            similarity_numbers[country][other_country]=number_of_values
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+queries[x]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+queries[x]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    num_cluster=6
    from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
    
    data_matrix = similarity_df
    model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
    
    
    cluster_table = pd.DataFrame(columns=["cluster","countries"])
    print("Clusters for "+queries[x])
    print(model.labels_)
    cluster_array=model.labels_
    for i in range (num_cluster):
        cluster=i
        cluster_name="Cluster "+str(cluster)
        array=[]
        for ind in range (len(countries)):
            if cluster_array[ind]==cluster:
                array.append(countries[ind])
        if len(array)!=0:
             df2 = {"cluster":cluster_name,'countries': array}
             cluster_table = cluster_table.append(df2, ignore_index = True)
        print("Cluster "+queries[x],cluster,": ",array)
    print("----------------------------")



    fig = ff.create_table(cluster_table)
    
    fig.update_layout(
    autosize=True,
    
    )
    
    fig.write_image("Clusters "+queries[x]+".png", scale=2)


fig, axs = plt.subplots(3,2,figsize=(30, 48),tight_layout=True)
plt.subplots_adjust(hspace=0.5) 
for a in range(3):
    for b in range(2):
        
        x=countries
        y=table[queries[a+b]].to_numpy()
            
            
            
        axs[a,b].bar(x, y,color=colors)
        
        
        axs[a,b].set_title("Average "+titles[a+b]+" Between 2015-2020",size=18)
        axs[a,b].annotate("\n"+notes[a+b]+"\n",
                xy = (0,-0.0245),
                xycoords='axes fraction',ha='left',va="center",fontsize=12)
        
        
text=notes[0]+"\n"+notes[1]+"\n"+notes[2]+"\n"+notes[3]+"\n"+notes[4]+"\n"+notes[5]+"\n"


plt.savefig("income_inequality"+".jpg", dpi=500)
plt.show()