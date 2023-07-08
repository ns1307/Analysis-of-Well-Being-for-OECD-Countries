import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import plotly.figure_factory as ff

dataframe= pd.read_csv("work_life_balance.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]


filtered_df = dataframe.loc[(dataframe['ISC11A']=='T')]
filtered_df = filtered_df.loc[(filtered_df['AGE']=="Y25T64")]
filtered_df = filtered_df.loc[(filtered_df['MEASURE']=="VALUE")]
queries=["EQLS_PERC_NEG_FAMILY","EQLS_PERC_NEG_JOB",
         "PIAAC_HOURS_WORK_WEEK","PIAAC_PERC_FLEX_WORK",
         
    
         
    ]
short_titles=["Difficult For Them To Concentrate At Work","Difficult For Them To Fulfil Their Family Responsibilities",
              "Mean Number Of Hours Worked Per Week","High or Very High Flexibility Of Working Hours"]
titles=["Adults Who Reported That Over The Last 12 Months It Has Been Difficult For Them To Concentrate At Work \nBecause Of Their Family Responsibilities",
        "Adults Who Reported That Over The Last 12 Months It Has Been Difficult For Them To Fulfil Their Family \nResponsibilities Because Of The Amount Of Time They Spend At Work ",
        "Mean Number Of Hours Worked Per Week In The Main Job Among Employed Adults",
        "Employed Adults Who Report Having A High or Very High Flexibility Of Working Hours \nIn Their Main Job"]
measure=["Percentage in Adults (%)","Percentage in Adults (%)",
         "Average Working Hours per Week (hours)","Percentage in Employed Adults (%)"]
table = pd.DataFrame(index=countries, columns=queries)

def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df

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


for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = dataframe.loc[(dataframe['COUNTRY']==countries[i])]#all data beloning to that country
        if not countr_filtered_df.empty:
            
                family=getFromFiltered(countr_filtered_df,"INDICATOR",queries[0])
                neg_job=getFromFiltered(countr_filtered_df,"INDICATOR",queries[1])
                hours_week=getFromFiltered(countr_filtered_df,"INDICATOR",queries[2])
                work_flex=getFromFiltered(countr_filtered_df,"INDICATOR",queries[3])
    
                if family.empty:
                    family_value=None
                else:
                    family_value=family.Value.to_numpy()[0]
                    table[queries[0]][country]=family_value
                if neg_job.empty:
                    upp_sec_value=None
                else:
                    upp_sec_value=neg_job.Value.to_numpy()[0]
                    table[queries[1]][country]=upp_sec_value
                if hours_week.empty:
                    hours_week_value=None
                else:
                    hours_week_value=hours_week.Value.to_numpy()[0]
                    table[queries[2]][country]=hours_week_value
                if work_flex.empty:
                    work_flex_value=None
                else:
                    work_flex_value=work_flex.Value.to_numpy()[0]
                    table[queries[3]][country]=work_flex_value
                    
                

for i in range (len(queries)):
    query=queries[i]
    non_NA_countries=get_non_na_countries(table, query)
    similarity_df=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=non_NA_countries, columns=non_NA_countries,dtype=np.float64)
    for country in non_NA_countries:
        value1=table.loc[country][query]
        for other_country in non_NA_countries:
            value2=table.loc[other_country][query]
            similarity, number_of_values = check_similarity(value1,value2)
            similarity_df[country][other_country]=similarity
            similarity_numbers[country][other_country]=number_of_values
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+short_titles[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+short_titles[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    num_cluster=6
    from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
    
    data_matrix = similarity_df
    model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
    
    cluster_table = pd.DataFrame(columns=["cluster","countries"])
    print("Clusters for "+short_titles[i])
    print(model.labels_)
    cluster_array=model.labels_
    clusters=[]
    for a in range (num_cluster):
        cluster=a
        cluster_name="Cluster "+str(cluster)
        array=[]
        for ind in range (len(non_NA_countries)):
            if cluster_array[ind]==cluster:
                array.append(non_NA_countries[ind])
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

    fig.write_image("Clusters-"+short_titles[i]+".png", scale=2)














fig, axs = plt.subplots(4,figsize=(10,20))
plt.subplots_adjust(hspace=0.5) 
title="Work Life Balance"
fig.suptitle(title)
for i in range(len(queries)):
    query=queries[i]
    values=[]
    country_names=[]
    new_colors=[]
    for j in range(len(countries)):
            country=countries[j]
            color=colors[j]
            nan = table.at[country,query]
            if nan == nan:
                country_names.append(country)
                
                value=table[query][country]
                values.append(value)
                new_colors.append(color)
    width = 0.5
    X_axis = np.arange(len(country_names))
    axs[i].bar( country_names , values, width,  edgecolor='black', color=new_colors)

    axs[i].set_title("\n"+titles[i])

    axs[i].set(xlabel='Countries', ylabel=measure[i])

plt.tight_layout()

plt.savefig("work_life_balance"+".jpg", dpi=500)
plt.show()
    