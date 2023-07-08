import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from numpy.linalg import norm
dataframe= pd.read_csv("social_spending.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]




table = pd.DataFrame(index=countries, columns=["value"])

new_countries=[]
no_data_countries=[]
new_colors= []
for i in range(len(countries)):
    country=countries[i]
    filtered_df = dataframe.loc[(dataframe['LOCATION']==country)]#all data beloning to that country
    filtered_df = filtered_df.loc[(filtered_df['MEASURE']=="PC_GDP")]
    
    if not filtered_df.empty:
        percantage=filtered_df.Value.to_numpy()[0]
        table["value"][country]=percantage
        
        new_countries.append(country)
        new_colors.append(colors[i])
    else:
        table = table.drop(country)
        no_data_countries.append(country)

countries=new_countries
colors=new_colors


def check_similarity(value1,value2):
    
    #similarity is 1 if they are equal, It can go to 0 as the values of numbers differ
    similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
    return similarity,1



similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
for country in countries:
    value1=table.loc[country]["value"]
    for other_country in countries:
        value2=table.loc[other_country]["value"]
        similarity, number_of_values = check_similarity(value1,value2)
        similarity_df[country][other_country]=similarity
        similarity_numbers[country][other_country]=number_of_values
import seaborn as sns 
colormap = sns.color_palette("coolwarm", 50)
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")

title="Similarity Matrix"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)


title="Number of Comparisons"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


num_cluster=6
from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering

data_matrix = similarity_df
model = AffinityPropagation(affinity='precomputed').fit(data_matrix)

cluster_table = pd.DataFrame(columns=["cluster","countries"])
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
    print("Cluster ",cluster,": ",array)



import plotly.figure_factory as ff

fig = ff.create_table(cluster_table)

fig.update_layout(
autosize=True,

)

fig.write_image("Clusters"+".png", scale=2)





values=table["value"].tolist()

fig = plt.figure()
ax = fig.add_subplot(111)


title="Percentage of Social Spending on GDP(%)"

plt.bar(countries, values, color=colors,
        width = 0.4)
 
plt.xlabel("Countries")
plt.ylabel("GDP (%)")
plt.title(title)

text="No data for: "+', '.join(no_data_countries)
plt.figtext(0,0, text, ha="left", fontsize=10)
plt.tight_layout()


plt.savefig(title+".png", dpi=500)
plt.show()