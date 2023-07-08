import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("skill_over_job_quality.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
perma_countries=countries.copy()
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]
perma_colors=colors.copy()
cluster_colors=[["orange","black"],["darkslategrey","black"],
                ["red","black"],["pink","black"],
                ["blue","black"],["green","grey"]]
skill_titles=["Low Skilled People","Medium Skilled People","High Skilled People","Total"]
skills=["LS","MS","HS","TP"]
titles=[["Earnings Quality in Dollars (in constant prices, at constant PPPs)","USD ($)"],["Quality of The Working Environment in Percentage","Percentage (%)"],
        ["Labour Market Insecurity in Percentage","Percentage (%)"]]
queries=["EQ","QWE","LMI"]
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df
def empty_list_check (arr1):
    i=0
    empty=True
    while empty and i<len(arr1):
        if arr1[i]!=None:
            empty=False
        i=i+1
    return empty

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
    avg_sim = sum(sim_array)/number_of_values
    #cosine=spatial.distance.cosine(new_array1, new_array2)
    return avg_sim,number_of_values
def get_non_na_countries(countries,table, index):
    non_na_countries=[]
    for country in countries:
        array=get_all_by_cou(table,country, index)
        empty=True
        i=0
        while empty and i<len(array):
            value=array[i]
            if value==value and value!=None:
                empty=False
            i=i+1
        if not empty:
            non_na_countries.append(country)
        
    return non_na_countries

def fill_na(values_list):
     index_not_na=[]
     for ind in range(len(values_list)):
         value=values_list[ind]
         if value!=None:
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
def get_all_by_cou(table,country, index):
    values_filtered=[]
    for year in years:
        
        values=table.loc[country][year]
    
    
        value=values[index]
        values_filtered.append(value)
        
    return values_filtered
    
years=list(np.arange(2005,2017))

for x in range(len(queries)):
    countries=perma_countries
    colors=perma_colors
    new_countries=[]
    no_data_countries=[]
    new_colors=[]
    table = pd.DataFrame(index=countries, columns=years)
    cluster_table=pd.DataFrame(index=["Cluster 0","Cluster 1","Cluster 2",
                                      "Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7"], 
                               columns=years)
    cluster_names=[]
    values=[]
    for i in range(len(countries)):
        country=countries[i]
        color=colors[i]
        countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==country)]#all data beloning to that country
        country_df=getFromFiltered(countr_filtered_df,"MEA",queries[x])  
        if not country_df.empty:
            new_countries.append(country)
            new_colors.append(color)
            for year in years:
                year_df=country_df.loc[(country_df['TIME']==year)]
                if not year_df.empty:
                    
                    ls_df = year_df.loc[(year_df['EDU']=="LS")]
                    if not ls_df.empty:
                        LS=ls_df.Value.to_numpy()[0]
                    else: 
                        LS=None
                        
                        
                    ms_df = year_df.loc[(year_df['EDU']=="MS")]
                    if not ms_df.empty:
                        MS=ms_df.Value.to_numpy()[0]
                    else:
                        MS=None
                    
                    hs_df = year_df.loc[(year_df['EDU']=="HS")]
                    if not hs_df.empty:
                        HS=hs_df.Value.to_numpy()[0]
                    else:
                        HS=None
                    
                    tp_df = year_df.loc[(year_df['EDU']=="TP")]
                    if not tp_df.empty:
                        TP=tp_df.Value.to_numpy()[0]
                    else:
                        TP=None
                    
                    
                    value=[LS,MS,HS,TP]
                    table[year][country]=value
                else:
                    table[year][country]=[None,None,None,None]
        else:
            table = table.drop(country)
            no_data_countries.append(country)
            
            
    countries=new_countries
    colors=new_colors+["black","black","black","black","black","black"]
    
    
    temp_countries=countries.copy()
    for a in range (len(skills)):
        non_NA_countries=get_non_na_countries(countries,table, a)
        similarity_df=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
        similarity_numbers=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
        for country in non_NA_countries:
            values1=get_all_by_cou(table,country,a)
            values1=fill_na(values1)
            countries_copy=non_NA_countries.copy()
            for other_country in countries_copy:
                values2=get_all_by_cou(table,other_country,a)
                values2=fill_na(values2)
                similarity, number_of_values = check_similarity(values1,values2)
                if similarity==0 or similarity != similarity:
                    if empty_list_check(values1):
                        similarity_df = similarity_df.drop(country)
                        similarity_df.drop([country], axis=1)
                        non_NA_countries.remove(country)
                    if empty_list_check(values2):
                        similarity_df = similarity_df.drop(other_country)
                        similarity_df=similarity_df.drop([other_country], axis=1)
                        non_NA_countries.remove(other_country)
                else:
                    similarity_df[country][other_country]=similarity
                    similarity_numbers[country][other_country]=number_of_values
        import seaborn as sns 
        colormap = sns.color_palette("coolwarm", 50)
        fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
        sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
        
        title="Similarity Matrix for "+ skills[a]+" in "+titles[x][0]
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig(title+".png", dpi=500)
        
        
        fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
        sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
        
        
        title="Number of Comparisons for "+ skills[a]+" in "+titles[x][0]
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig(title+".png", dpi=500)
        
        
        num_cluster=8
        from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
        
        data_matrix = similarity_df
        model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
        
        cluster_out_table = pd.DataFrame(columns=["cluster","countries"])
        print(model.labels_)
        cluster_array=model.labels_
        clusters=[]
        print("Clusters for "+ skills[a]+" in "+titles[x][0])
        for i in range (num_cluster):
            cluster=i
            cluster_name="Cluster "+str(cluster)
            array=[]
            for ind in range (len(non_NA_countries)):
                if cluster_array[ind]==cluster:
                    array.append(non_NA_countries[ind])
            if len(array)!=0:
                df2 = {"cluster":cluster_name,'countries': array}
                cluster_out_table = cluster_out_table.append(df2, ignore_index = True)
            clusters.append(array)
            print("Cluster ",cluster,": ",array)
        print("------------------------------")
        fig = ff.create_table(cluster_out_table)

        fig.update_layout(
        autosize=True,

        )

        fig.write_image("Clusters "+ skills[a]+" in "+titles[x][0]+".png", scale=2)

        for i  in range (num_cluster):
            cluster=clusters[i]
            cluster_name="Cluster "+str(i)
            if(len(cluster)>0):
                cluster_countries=cluster
                cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
                for year in years:
                    values=[]
                    for country in cluster_countries:
                        value=table[year][country][a]
                        if value!=None :
                            values.append(value)
                    if len(values)>0:
                        avg_value=sum(values)/len(values)
                    else: 
                        avg_value=None
                    
                    arr=cluster_table[year][cluster_name]
                    if arr!=arr:
                        arr=[]
                    arr.append(avg_value)
                    cluster_df[year][cluster_name]=arr
                if cluster_name in cluster_table.index:
                    cluster_table = cluster_table.drop(cluster_name)
                cluster_table=cluster_table.append(cluster_df)
                if not cluster_name in cluster_names:
                    cluster_names.append(cluster_name)
            else:#if cluster has no value to append table
                cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
                for year in years:
                    arr=cluster_table[year][cluster_name]
                    if arr!=arr:
                        arr=[]
                    arr.append(None)
                    cluster_df[year][cluster_name]=arr
                if cluster_name in cluster_table.index:
                    cluster_table = cluster_table.drop(cluster_name)
                cluster_table=cluster_table.append(cluster_df)

    table=table.append(cluster_table)
        






    
    j=0
    
    fig, axs = plt.subplots(4,figsize=(6,20))
    plt.subplots_adjust(hspace=0.5) 
    title=titles[x][0]
    fig.suptitle(title)
    table_countries=table.index
    for a in range (len(skills)):
            c=0
            for j in range(len(table_countries)):
               skill=skills[a]
               country=table_countries[j]
               
               values=table.loc[country].to_numpy()
               
               years_filtered=[]
               values_filtered=[]
               for b  in range(len(values)):
                   if values[b]!=None:#if not null
                       value=values[b][a]
                       if value!=None:
                           years_filtered.append(years[b])
                           
                           values_filtered.append(value)
               X=years_filtered
               y=values_filtered
               if len(y)>0:
                   if("Cluster" in country):
                       axs[a].plot(X,y,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[c][0] ,gapcolor=cluster_colors[c][1])
                       c=c+1
                   else:
                       axs[a].plot(X,y,".-",lw=1.5,markersize=12,label=country,c=colors[j])
                   
                   axs[a].set_title("\n"+skills[a])
                   axs[a].set(xlabel='Year', ylabel=titles[x][1])
                   axs[a].legend(prop={'size': 5})
    if len(no_data_countries)!=0:
        text="No data for: "+', '.join(no_data_countries)
        plt.figtext(0,0, text, ha="left",fontweight="bold", fontsize=7)
       
    plt.tight_layout()
        
    plt.savefig(title+".jpg", dpi=500)
    plt.show()