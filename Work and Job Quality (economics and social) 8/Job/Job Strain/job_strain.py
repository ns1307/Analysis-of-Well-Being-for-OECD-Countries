import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("job_strain.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]

colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]
cluster_colors=[["orange","black"],["darkslategrey","black"],
                ["red","black"],["pink","black"],
                ["blue","black"],["green","grey"]]
fig_titles=["High Level of Job Demands","Low Level of Job Resources"]

job_demand_queries=["1_1_1","1_1_2","1_1_3"]

job_resources_queries=["1_2_1","1_2_2","1_2_3"]

job_demand_titles=["Physical Health Risk Factors",
                   "Long Working Hours",
                   "Inflexibility of Working Hours",
                   ]
job_resources_titles=["Work Autonomy and Learning Opportunities",
                      "Training and Learning",
                      "Opportunity for Career Advancement"]

titles=[job_demand_titles,job_resources_titles]


years=list(np.arange(2005,2018,5))
job_strain_table = pd.DataFrame(index=countries, columns=years)
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df
def getQueriesFromdDF(df, var_name , queries):
    values=[]
    for i in range (len(queries)):
        
        value_df=df.loc[(df[var_name]==queries[i])]
        if value_df.empty:
            value=None
        else:
            value=value_df.Value.to_numpy()[0]
        values.append(value)
    return values

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
def get_all_by_cou(table,country, index1, index2):
    values_filtered=[]
    for year in years:
        
        values=table.loc[country][year]
    
    
        value=values[index1][index2]
        values_filtered.append(value)
        
    return values_filtered







dataframe=dataframe.loc[(dataframe["POP"]=="TP")]
dataframe=dataframe.loc[(dataframe["SEX"]=="TP")]
dataframe=dataframe.loc[(dataframe["EDU"]=="TP")]
no_data_countries=[]
new_colors=[]
new_countries=[]
for i in range(len(countries)):
        country=countries[i]
        color=colors[i]
        countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==countries[i])]#all data beloning to that country
        if not countr_filtered_df.empty:
            for year in years:
                year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==year)]
                demand_values= getQueriesFromdDF(year_df, "VAR", job_demand_queries)
                resource_values=getQueriesFromdDF(year_df, "VAR", job_resources_queries)
                values=[demand_values,resource_values]
                job_strain_table[year][country]=values
            new_countries.append(country)
            new_colors.append(color)
        else:
                job_strain_table = job_strain_table.drop(country)
                no_data_countries.append(country)
countries=new_countries
colors=new_colors+["black","black","black","black","black","black"]









cluster_names=[]
cluster_table=pd.DataFrame(index=["Cluster 0","Cluster 1","Cluster 2",
                                  "Cluster 3","Cluster 4","Cluster 5"], 
                           columns=years)
temp_countries=countries.copy()
for index1 in range(2):
    for index2 in range(3):
        countries=temp_countries
        similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
        similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
        for country in countries:
            values1=get_all_by_cou(job_strain_table,country,index1,index2)
            values1=fill_na(values1)
            countries_copy=countries.copy()
            for other_country in countries_copy:
                values2=get_all_by_cou(job_strain_table,other_country,index1,index2)
                values2=fill_na(values2)
                similarity, number_of_values = check_similarity(values1,values2)
                if similarity==0 or similarity != similarity:
                    if empty_list_check(values1):
                        similarity_df = similarity_df.drop(country)
                        similarity_df.drop([country], axis=1)
                        countries.remove(country)
                    if empty_list_check(values2):
                        similarity_df = similarity_df.drop(other_country)
                        similarity_df=similarity_df.drop([other_country], axis=1)
                        countries.remove(other_country)
                else:
                    similarity_df[country][other_country]=similarity
                    similarity_numbers[country][other_country]=number_of_values
        import seaborn as sns 
        colormap = sns.color_palette("coolwarm", 50)
        fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
        sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
        
        title="Similarity Matrix for "+titles[index1][index2]+" in "+fig_titles[index1]
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig(title+".png", dpi=500)
        
        
        fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
        sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
        
        
        title="Number of Comparisons for "+titles[index1][index2]+" in "+fig_titles[index1]
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig(title+".png", dpi=500)
        
        
        num_cluster=6
        from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
        
        data_matrix = similarity_df
        model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
        
        cluster_out_table = pd.DataFrame(columns=["cluster","countries"])
        print(model.labels_)
        cluster_array=model.labels_
        clusters=[]
        print("Clusters for "+titles[index1][index2]+" in "+fig_titles[index1])
        for i in range (num_cluster):
            cluster=i
            cluster_name="Cluster "+str(cluster)
            array=[]
            for ind in range (len(countries)):
                if cluster_array[ind]==cluster:
                    array.append(countries[ind])
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

        fig.write_image("Clusters "+titles[index1][index2]+" in "+fig_titles[index1]+".png", scale=2)
        for i  in range (num_cluster):
            cluster=clusters[i]
            cluster_name="Cluster "+str(i)
            if(len(cluster)>0):
                cluster_countries=cluster
                cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
                for year in years:
                    values=[]
                    for country in cluster_countries:
                        value=job_strain_table[year][country][index1][index2]
                        if value!=None :
                            values.append(value)
                    if len(values)>0:
                        avg_value=sum(values)/len(values)
                    else: 
                        avg_value=None
                    
                    arr=cluster_table[year][cluster_name]
                    if arr!=arr:
                        arr=[[],[]]
                    arr[index1].append(avg_value)
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
                        arr=[[],[]]
                    arr[index1].append(None)
                    cluster_df[year][cluster_name]=arr
                if cluster_name in cluster_table.index:
                    cluster_table = cluster_table.drop(cluster_name)
                cluster_table=cluster_table.append(cluster_df)
countries=temp_countries+cluster_names
job_strain_table=job_strain_table.append(cluster_table)
        















for x in range(2):
    fig, axs = plt.subplots(3,figsize=(6,20))
    plt.subplots_adjust(hspace=0.5) 
    title =fig_titles[x]
    fig.suptitle(title)
    
    for a in range (3):
        c=0
        for i in range(len(countries)):
            country=countries[i]
            color=colors[i]
            values=[]
            years_filtered=[]
            for j in range (len(years)):
                year=years[j]
                value=job_strain_table[year][country][x][a]
                if value!=None and value==value:
                    values.append(value)
                    years_filtered.append(year)

                
            
            X=years_filtered
            y=values
            best_degree=11
            """
            best_degree=i
            best_score=0
            for i in range(5,18,1):
                mymodel = np.poly1d(np.polyfit(X, tot_unemp_values, i))
                myline = np.linspace(years[0]-2,years[-1]+2)
                
                score=r2_score(tot_unemp_values, mymodel(X))
                if score>best_score:
                    best_degree=i
                    best_score=score
            print(country,"---degree ", best_degree,"--score: ",best_score)
        
            mymodel = np.poly1d(np.polyfit(X, avg_unemp_values , best_degree))
            myline = np.linspace(years[0]-1,years[-1]+2)
            """
            
            if("Cluster" in country):
                axs[a].plot(X,y,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[c][0] ,gapcolor=cluster_colors[c][1])
                c=c+1
            else:
                axs[a].plot(X,y,".-",lw=1.5,markersize=12,label=country,c=color)
            axs[a].set_title("\n"+titles[x][a])
            axs[a].set(xlabel='Year', ylabel="Percentage (%)")
            axs[a].legend(countries,prop={'size': 5})
            
        
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)
    
        
                