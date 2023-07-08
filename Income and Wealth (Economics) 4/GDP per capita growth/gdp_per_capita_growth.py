import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm


dataframe= pd.read_csv("gdp_per_capita_growth.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]

filtered_df=dataframe.filter(countries)


years=list(np.arange(2000,2022))
table = pd.DataFrame(index=countries, columns=years)
filtered_df = dataframe.loc[(dataframe['SUBJECT']=="T_GDPPOP_V")]
for i in range(len(countries)):
    country=countries[i]
    country_df = filtered_df.loc[(filtered_df['LOCATION']==country)]
    for year in years:
        year_df=country_df.loc[(country_df['TIME']==year)]
        if not year_df.empty:
            percantage=year_df.Value.to_numpy()[0]
            table[year][country]=percantage


 
 
def check_similarity(array1,array2):
    number_of_values=0
    new_array1=[]
    new_array2=[]
    for ind in range(len(array1)):
        value1=array1[ind]
        value2=array2[ind]
        if value1==value1 and value2==value2:#if values are not empty
            new_array1.append(value1)
            new_array2.append(value2)
            number_of_values=number_of_values+1
    cosine = np.dot(new_array1,new_array2)/(norm(new_array1)*norm(new_array2))
    #cosine=spatial.distance.cosine(new_array1, new_array2)
    return cosine,number_of_values

similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
for country in countries:
    values1=table.loc[country].to_numpy().astype(float)
    for other_country in countries:
        values2=table.loc[other_country].to_numpy().astype(float)
        similarity, number_of_values = check_similarity(values1,values2)
        similarity_df[country][other_country]=similarity
        similarity_numbers[country][other_country]=number_of_values
import seaborn as sns 
colormap = sns.color_palette("coolwarm", 50)
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")

title="Similarity Matrix Room per individual"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)


title="Number of Comparisons Room per individual"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


num_cluster=6
from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering

data_matrix = similarity_df
model = AffinityPropagation(affinity='precomputed').fit(data_matrix)





for i in range(len(countries)):
    years=years
    country=countries[i]
    color=colors[i]
    values=table.loc[country].to_numpy().astype(float)
    
    j=0
    
    X=years
    y=values
    best_degree=1 #below comment line is tried, best result is with 6th degree
    
    best_score=0
    for a in range(10,80,1):
        mymodel = np.poly1d(np.polyfit(X, y, a))
        myline = np.linspace(years[0]-2,years[-1]+2)
        
        score=r2_score(y, mymodel(X))
        if score>best_score:
            best_degree=a
            best_score=score
            

    mymodel = np.poly1d(np.polyfit(X, y, best_degree))
    myline = np.linspace(years[0]-1,years[-1]+1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    plt.scatter(X, y,c="ORANGE")
    
    text="R2 score:"+str(round(r2_score(y, mymodel(X)),2))
    plt.figtext(0,0, text, ha="left", fontsize=10)
    
    plt.plot(myline, mymodel(myline),c=color)
    
    plt.legend(["actual values","GDP per capita prediction"])
    title="GDP per Capita Growth Chart for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Percantage of GDP Growth by year(%)")
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)



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
