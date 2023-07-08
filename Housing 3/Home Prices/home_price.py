import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff

dataframe= pd.read_csv("home_price.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]
perma_countries=countries.copy()
perma_colors=colors.copy()
cluster_colors=[["darkslategrey","black"],["red","black"],["blue","orange"],
                    ["pink","black"],
                    ["orange","black"],["green","grey"]]
time=["2018-Q2","2018-Q3","2018-Q4",
      "2019-Q1","2019-Q2","2019-Q3","2019-Q4",
      "2020-Q1","2020-Q2","2020-Q3","2020-Q4",
      "2021-Q1","2021-Q2","2021-Q3","2021-Q4",
      "2022-Q1","2022-Q2","2022-Q3","2022-Q4",
      "2023-Q1"]
measures=["Percentage change from previous period","Percentage change on the same period of the previous year","Index publication base"]
filtered_df=dataframe.filter(countries)

texts=[" (difference between quarters)"," (change from same quarter last year)"," (The price change of the house, which was 100 dollars in 2015)"]
  

filtered_df = dataframe.loc[(dataframe['VAR']=="RHPI")]
filtered_df = filtered_df.loc[(filtered_df['DWELLINGS']=="DWELLINGS_TOTAL")]
filtered_df = filtered_df.loc[(filtered_df['VINTAGE']=="VINTAGE_TOTAL")]





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
     sim_array=[]
     for ind in range(len(array1)):
         value1=array1[ind]
         value2=array2[ind]
         if value1==value1 and value2==value2 and value1!=None and value2!=None:
             number_of_values=number_of_values+1
             if value1==0 and value2==0:
                 similarity=1
             else :
                 r1=abs(value1 - value2)
                 r2=abs(value1 + value2)
                 if r1>r2:
                     temp=r1
                     r1=r2
                     r2=temp
                 similarity= 1 - (r1 / r2)
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
             #ratio of small to big
             avg_sim= 1 - abs(abs(avg1 - avg2) / (avg1 + avg2))
     return avg_sim,number_of_values

for i in range(3):
    countries=perma_countries.copy()
    colors=perma_colors.copy()

    table = pd.DataFrame(index=countries, columns=time)

    new_countries=[]
    no_data_countries=[]
    new_colors=[]
    for cou_no in range(len(countries)):
        country=countries[cou_no]
        
        country_df = filtered_df.loc[(filtered_df['REG_ID']==country)]
        measure_df = country_df.loc[(country_df['Measure']==measures[i])]
        if measure_df.empty:
            table = table.drop(country)
            no_data_countries.append(country)
        else:
            new_countries.append(country)
            new_colors.append(colors[cou_no])
            for quarter in time:
                time_df=measure_df.loc[(measure_df['TIME']==quarter)]
                if time_df.empty:
                    percantage=None
                else:
                    percantage=time_df.Value.to_numpy()[0]
                    table[quarter][country]=percantage
                
            
    countries=new_countries
    colors=new_colors+["black","black","black","black","black","black","black"]

    
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
    
    title="Similarity Matrix for "+measures[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+measures[i]
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
    for q in range (num_cluster):
        cluster=q
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
    
    fig.write_image("Clusters "+measures[i]+".png", scale=2)

    for cluster_no  in range (num_cluster):
            cluster=clusters[cluster_no]
            cluster_name="Cluster "+str(cluster_no)
            if(len(cluster)>0):
                cluster_countries=cluster
                cluster_df = pd.DataFrame(index=[cluster_name], columns=time)
                for quarter in time:
                    values=[]
                    for country in cluster_countries:
                        value=table[quarter][country]
                        if value==value and value!=None:
                            values.append(value)
                    if len(values)!=0:
                        avg_value=sum(values)/len(values)
                        cluster_df[quarter][cluster_name]=avg_value
                table=table.append(cluster_df)
                countries.append(cluster_name)




    c=0
    plt.figure(figsize=(25,15))
    for cou_no in range (len(countries)):
        country=countries[cou_no]
        color=colors[cou_no]
        quarters=[]
        values=[]
        
        for quarter in time:
            value =table[quarter][country]
            
            if value!=None:
                values.append(value)
                quarters.append(quarter)
            
            
        X=quarters
        y=values
        if("Cluster" in country):
            plt.plot(X, y,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[c][0] ,gapcolor=cluster_colors[c][1])
            c=c+1
        else:
            plt.plot( X, y,".-",label=country, c=color)
      
    
    plt.title(measures[i]+texts[i])
    plt.legend(loc="upper left")

    text="No data for: "+', '.join(no_data_countries)
    plt.figtext(0,0, text, ha="left", fontsize=14)
    
    plt.tight_layout()
    
    plt.savefig(measures[i]+".jpg", dpi=500)
    plt.show()