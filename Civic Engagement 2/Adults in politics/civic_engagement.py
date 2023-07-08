import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from numpy.linalg import norm


dataframe= pd.read_csv("civic_engagement.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
education=['L0T2','L3T4','L5T8','T']
education_str=["Below upper secondary","Upper secondary, post-secondary","Tertiary","Total"]
filtered_df=dataframe.filter(countries)
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue"]


table = pd.DataFrame(index=countries, columns=education_str)
indicators=["ESS_ISSP_PERC_POL_INTEREST","ESS_ISSP_PERC_POL_EFFICACY"]
short_title=["Adults interested in politics by edu","Adults that political system allows having a say"]
title_list=["Percentage of adults who are being interested in politics, by educational attainment",
            "Percentage of adults that the political system allows having a say in what the government does"]
def check_similarity(value1,value2):
    
    #similarity is 1 if they are equal, It can go to 0 as the values of numbers differ
    if value1==0 and value2==0:
        similarity=1
    else:
        similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
    return similarity,1
def get_non_na_countries(table, query):
    non_na_countries=[]
    for country in countries:
        value=table.loc[country][query]
        if value !=None and value==value:
            non_na_countries.append(country)
    return non_na_countries

no_data_countries=[]
for q in range(len(indicators)): 
    filtered_df = dataframe.loc[(dataframe['INDICATOR']==indicators[q])]
    
    
    new_countries=[]
    new_colors=[]
    for j in range(len(countries)):
        country=countries[j]
        country_df = filtered_df.loc[(filtered_df['COUNTRY']==country)]#all data beloning to that country
        country_df = country_df.loc[(filtered_df['MEASURE']=="VALUE")]
        
        if not country_df.empty:
            for i in range(len(education)):
                ed_df=country_df.loc[(country_df['ISC11A']==education[i])]
        
                total=ed_df.loc[(ed_df['AGE']=="Y25T64")]
                if total.empty:
                    value=np.nan
                else:
                    value=ed_df["Value"].values[0]
                table[education_str[i]][country]=value
            new_countries.append(country)
            new_colors.append(colors[j])
        else:
            table = table.drop(country)
            no_data_countries.append(country)
    
    countries=new_countries
    colors=new_colors



    
    """     
    def check_similarity(array1,array2):
        number_of_values=0
        ratio_array=[]
        new_array1=[]
        new_array2=[]
        for ind in range(len(array1)):
            value1=array1[ind]
            value2=array2[ind]
            if value1==value1 and value2==value2:#if values are not empty
                diff=abs(value1-value2)
                ratio=diff/max(value1,value2)
                ratio_array.append(ratio)
                number_of_values=number_of_values+1
        cosine =sum(ratio_array)/number_of_values
        return cosine,number_of_values"""
    
    for i in range(len(education_str)):
        edu_level=education_str[i]
        non_NA_countries=get_non_na_countries(table, edu_level)
        similarity_df=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
        similarity_numbers=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
        for country in non_NA_countries:
            value1=table[edu_level][country]
            for other_country in non_NA_countries:
                value2=table[edu_level][other_country]
                similarity, number_of_values = check_similarity(value1,value2)
                similarity_df[country][other_country]=similarity
                similarity_numbers[country][other_country]=number_of_values
        import seaborn as sns 
        colormap = sns.color_palette("coolwarm", 50)
        fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
        sns.heatmap(similarity_df, annot=True, linewidths=.5, ax=ax,cmap=colormap,fmt=".3f")
        
        title="Similarity Matrix for table : \n "+short_title[q]+"-"+edu_level
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig("Similarity Matrix "+short_title[q]+edu_level+".png", dpi=500)
        
        
        fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
        sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
        
        
        title="Number of Comparisons for table : \n "+short_title[q]+"-"+edu_level
        plt.title(title)
        plt.xlabel("Countries")
        plt.ylabel("Countries")
        plt.savefig("Number of Comparisons "+short_title[q]+"-"+edu_level+".png", dpi=500)
        
        
        num_cluster=5
        from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
        
        data_matrix = similarity_df
        model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
        cluster_table = pd.DataFrame(columns=["cluster","countries"])
        print("Clusters for : "+short_title[q]+"-"+edu_level)
        print(model.labels_)
        cluster_array=model.labels_
        for i in range (num_cluster):
            cluster=i
            cluster_name="Cluster "+str(cluster)
            array=[]
            for ind in range (len(non_NA_countries)):
                if cluster_array[ind]==cluster:
                    array.append(non_NA_countries[ind])
                df2 = {"cluster":cluster_name,'countries': array}
            if len(array)!=0:
                cluster_table = cluster_table.append(df2, ignore_index = True)
            print("Cluster ",cluster,": ",array)
            
        import plotly.figure_factory as ff
    
        fig = ff.create_table(cluster_table)
        
        fig.update_layout(
        autosize=True,
        
        )
        
        fig.write_image("Clusters_"+short_title[q]+"-"+edu_level+".png", scale=2)
        
    
    subplots=4
    
    fig, axs = plt.subplots(subplots,figsize=(15, 30))
    
    for x in range (len(countries)):
        country=countries[x]
        color=colors[x]
        for i in range (subplots):
            y1=table[education_str[i]][country]
            x1=country
            if y1==y1:
                axs[i].bar(x1, y1,label=country,color=color)
                
    
    for i in range (subplots):
            
        axs[i].set_title(education_str[i], size=20)
        axs[i].legend(loc="upper left")
    
        axs[i].set_xlabel('Percentage', fontsize = 15)
        axs[i].set_ylabel('Countries', fontsize = 15)
    
    text="No data for: "+', '.join(no_data_countries)
    plt.figtext(0,0, text, ha="left",fontweight="bold", fontsize=17)
    

    fig.suptitle(title_list[q],fontweight="bold", fontsize=18.5)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.954)
    plt.savefig(title_list[q]+".png", dpi=500)
    plt.show()
    
