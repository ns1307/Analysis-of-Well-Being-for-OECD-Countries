import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("traffic_accident_rate.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
perma_countries=countries.copy()
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]
perma_colors=colors.copy()


queries=["IND-SAFE-KILL-VEH-KM","IND-SAFE-KILL-PC","IND-SAFE-KILL-VEH"]
titles=["Road Fatalities per 1.000.000 vehicle-km","Road Fatalities Rate by Inhabitants",
        "Road Fatalities Rate by Road Motor Vehicles"]
labels=["Road Fatalities Rate (per 1.000.000 vehicle-km)","Road Fatalities Rate (per 100.000 inhabitant)",
        "Road Fatalities Rate (per 10.000 road motor vehicles)"]
years=list(np.arange(2000,2020))
table = pd.DataFrame(index=countries, columns=years)
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df
def check_similarity(array1,array2):
     number_of_values=0
     sim_array=[]
     for ind in range(len(array1)):
         value1=array1[ind]
         value2=array2[ind]
         if value1==value1 and value2==value2 and value1!=None and value2!=None:
             number_of_values=number_of_values+1
             if value1==0 and value2==0:
                 similarity=1
             else :
                 similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
             sim_array.append(similarity)
     if number_of_values>0:
         avg_sim = sum(sim_array)/number_of_values
     #cosine=spatial.distance.cosine(new_array1, new_array2)
     else:
             new_array1=[]
             new_array2=[]
             for ind in range(len(array1)):
                 value1=array1[ind]
                 value2=array2[ind]
                 if value1==value1: #if values are not empty
                     new_array1.append(value1)
                 if  value2==value2:
                     new_array2.append(value2)
    
             avg1=sum(new_array1)/len(new_array1)
             avg2=sum(new_array2)/len(new_array2)
             avg_sim= 1 - abs(abs(avg1 - avg2) / (avg1 + avg2))
     return avg_sim,number_of_values
fig0, axs0 = plt.subplots(3,figsize=(10,20))
plt.subplots_adjust(hspace=0.5) 
title="Road Fatalities Data\n"
fig0.suptitle(title)

for q in range(3):
    countries=perma_countries
    colors=perma_colors
    new_countries=[]
    no_data_countries=[]
    table = pd.DataFrame(index=countries, columns=years)
    new_colors=[]
    for i in range(len(countries)):
            country=countries[i]
            countr_filtered_df = dataframe.loc[(dataframe['COUNTRY']==countries[i])]#all data beloning to that country
            country_df=getFromFiltered(countr_filtered_df,"INDICATOR",queries[q])
            if not country_df.empty:
                for year in years:
                    year_df=country_df.loc[(country_df['YEAR']==year)]
                    if not year_df.empty:
                        value=year_df.Value.to_numpy()[0]
                        table[year][country]=value
                new_countries.append(country)
                new_colors.append(colors[i])
            else:
                table= table.drop(country)
                no_data_countries.append(country)
                
    
    countries=new_countries
    colors=new_colors+["black","black","black","black","black","black"]
    
    
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
    
    title="Similarity Matrix for "+titles[q]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+titles[q]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    num_cluster=6
    from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
    
    data_matrix = similarity_df
    model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
    
    cluster_table = pd.DataFrame(columns=["cluster","countries"])
    cluster_array=model.labels_
    clusters=[]
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
        clusters.append(array)
        
    
    fig = ff.create_table(cluster_table)
    
    fig.update_layout(
    autosize=True,
    
    
    )
    
    fig.write_image("Clusters "+titles[q]+".png", scale=2)

    
    cluster_colors=[["darkslategrey","black"],["red","black"],["blue","orange"],
                    ["pink","black"],
                    ["orange","black"],["green","grey"]]
    for cluster_no  in range (num_cluster):
        cluster=clusters[cluster_no]
        cluster_name="Cluster "+str(cluster_no)
        if(len(cluster)>0):
            cluster_countries=cluster
            cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
            for year in years:
                values=[]
                for country in cluster_countries:
                    value=table[year][country]
                    if value==value:
                        values.append(value)
                if len(values)!=0:
                    avg_value=sum(values)/len(values)
                    cluster_df[year][cluster_name]=avg_value
            table=table.append(cluster_df)
            countries.append(cluster_name)







    
    print(model.labels_)
    print("Clusters for "+titles[q])
    for i in range (num_cluster):
        cluster=i
        array=clusters[i]
        print("Cluster ",cluster,": ",array)
    print("-----------------------")


    c=0
    for j in range(len(countries)):
        country=countries[j]
        color=colors[j]
        values=[]
        years_filtered=[]
        for i in range(len(years)):
            year=years[i]
            value=table[year][country]
            if value!=None:  
                values.append(value)
                years_filtered.append(year)
        if("Cluster" in country):
            axs0[q].plot(years_filtered,values,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[c][0] ,gapcolor=cluster_colors[c][1])
            c=c+1
        else:
            axs0[q].plot( years_filtered,values,".-",label=country, c=color)
        
    axs0[q].set_title(titles[q])
    
    axs0[q].set(xlabel='Year', ylabel=labels[q])
        
    axs0[q].legend(countries,prop={'size': 5})
    
    fig0.tight_layout()

    

fig0.savefig("road_fatalities_data"+".jpg", dpi=500)
fig0.show()