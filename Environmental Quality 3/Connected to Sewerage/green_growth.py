import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from numpy.linalg import norm


dataframe= pd.read_csv("green_growth.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]

filtered_df = dataframe.loc[(dataframe['VAR']=="ASEW_POP")]


years=list(np.arange(2000,2019))
table = pd.DataFrame(index=countries, columns=years)

no_data_countries=[]
new_countries=[]


for i in range(len(countries)):
    country=countries[i]
    
    country_df = filtered_df.loc[(filtered_df['COU']==country)]#all data beloning to that country
    if not country_df.empty:
        for year in years:
            year_df=country_df.loc[(country_df['YEA']==year)]
            if year_df.empty:
                percantage=None
            else:
    
                percantage=year_df.Value.to_numpy()[0]
            table[year][country]=percantage
        new_countries.append(country)
    else:
        table = table.drop(country)
        no_data_countries.append(country)
 
countries=new_countries
def fill_na(values_list):
     index_not_na=[]
     for ind in range(len(values_list)):
         value=values_list[ind]
         if value==value:
             index_not_na.append(ind)
     for ind in range(len(index_not_na)):
         if ind!=len(index_not_na)-1:
             first_ind=index_not_na[ind]
             last_ind=index_not_na[ind+1]
             gap=last_ind-first_ind-1
             if gap!=0:
                 first_val=values_list[first_ind]
                 last_val=values_list[last_ind]
                 diff=last_val-first_val
                 avg=diff/(gap+1)
                 for x in range(gap):
                     values_list[first_ind+1]=first_val+avg
                     first_ind=first_ind+1
                     first_val=values_list[first_ind]
     return values_list
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
    values1=fill_na(values1)
    for other_country in countries:
        values2=table.loc[other_country].to_numpy().astype(float)
        values2=fill_na(values2)
        similarity, number_of_values = check_similarity(values1,values2)
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


for i in range(len(countries)):
    country=countries[i]
    color=colors[i]
    
    values=table.loc[country].to_numpy().astype(float)
    
    years_filtered=[]
    values_filtered=[]
    for x  in range(len(values)):
        if values[x]==values[x]:
            years_filtered.append(years[x])
            values_filtered.append(values[x])
    
    X=years_filtered
    y=values_filtered
    
    best_degree=7 #below comment line is tried, best result is with 6th degree
    """best_degree=3
        best_score=0
        for i in range(5,10,1):
            mymodel = np.poly1d(np.polyfit(X, y, i))
            myline = np.linspace(years[0]-2,years[-1]+2)
            
            score=r2_score(y, mymodel(X))
            if score>best_score:
                best_degree=i
                best_score=score
        print("degree",best_degree,"score",best_score)    
     """
    mymodel = np.poly1d(np.polyfit(X, y, best_degree))
    myline = np.linspace(years[0],years[-1]+2)
       
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
       
    plt.scatter(X, y,c="ORANGE")
    plt.legend(["actual values"])
    text="R2 score:"+str(round(r2_score(y, mymodel(X)),2))+"\n"
    plt.figtext(0,0, text, ha="left", fontsize=10)
        #y ekseninin aralığını belirlemek için
        #plt.ylim([90, 100])
        #x ekseninin aralığını belirlemek için
        #plt.xlim([2000, 2020])
        
    plt.plot(myline, mymodel(myline), c=color)
    title="Population connected to public sewerage for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Population (%)")
    plt.tight_layout()
        #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)
    
        