import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import math

import plotly.figure_factory as ff


dataframe= pd.read_csv("social_connections.csv")

countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
perma_countries=countries.copy()
titles=["Percentage of adults who participated in any cultural or sporting activities in the last 12 months",
        "Percentage of adults who get together with friends living outside their household at least once a week",
        "Percentage of adults who have someone to ask for help",
        "Percentage of adults who actively participate in social media on a daily basis",
        "Percentage of adults who participated in formal voluntary activities in the last 12 months"]

queries=["EUSILC_PERC_CULT_SPORT","EUSILC_PERC_FRIENDS","EUSILC_PERC_HELP","EUSILC_PERC_SOCMEDIA",
         "EUSILC_PERC_VOLUNTEER"]
ed_levels=["L0T2","L3T4","L5T8","T"]
ed_titles=['Below upper secondary education','Upper secondary and post-secondary non-tertiary education',
        'Tertiary education','All levels of education']
table = pd.DataFrame(index=countries, columns=queries)
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df

def check_similarity(value1,value2):
    
    similarity= 1 - abs(abs(value1 - value2) / (value1 + value2))
    return similarity,1

fig0, axs0 = plt.subplots(5,figsize=(10,20))
plt.subplots_adjust(hspace=0.5) 
title="Socail Connections Data"
fig0.suptitle(title)
for q in range(len(queries)):
    countries=perma_countries
    no_data_countries=[]
    query=queries[q]
    table = pd.DataFrame(index=countries, columns=ed_levels)
    copy_countries=countries.copy()
    for i in range(len(copy_countries)):
        country=copy_countries[i]
        countr_filtered_df = dataframe.loc[(dataframe['COUNTRY']==country)]#all data beloning to that country
        if not countr_filtered_df.empty:
            
                category_df=getFromFiltered(countr_filtered_df,"INDICATOR",query)
                for a in range(len(ed_levels)):
                
                    edu_df=category_df.loc[(category_df['ISC11A']==ed_levels[a])]
                    if not edu_df.empty:
                        value=edu_df.Value.to_numpy()[0]
                        table[ed_levels[a]][country]=value
        else:
            table= table.drop(country)
            no_data_countries.append(country)
            countries.remove(country)






    for a in range(len(ed_levels)):    
            similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
            similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
            for country in countries:
                value1=table[ed_levels[a]][country]
                for other_country in countries:
                    value2=table[ed_levels[a]][other_country]
                    similarity, number_of_values = check_similarity(value1,value2)
                    similarity_df[country][other_country]=similarity
                    similarity_numbers[country][other_country]=number_of_values
            import seaborn as sns 
            colormap = sns.color_palette("coolwarm", 50)
            fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
            sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
            
            title="Similarity Matrix for "+ed_titles[a]+" "+titles[q]
            plt.title(title)
            plt.xlabel("Countries")
            plt.ylabel("Countries")
            plt.savefig(title+".png", dpi=500)
            
            
            fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
            sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
            
            
            title="Number of Comparisons for "+ed_titles[a]+" "+titles[q]
            plt.title(title)
            plt.xlabel("Countries")
            plt.ylabel("Countries")
            plt.savefig(title+".png", dpi=500)
            
            
            num_cluster=6
            from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
            
            data_matrix = similarity_df
            model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
            
            
            cluster_table = pd.DataFrame(columns=["cluster","countries"])

            print("Clusters for "+ed_titles[a]+" "+titles[q])
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
            print("-------------------------")
        

            fig = ff.create_table(cluster_table)
            
            fig.update_layout(
            autosize=True,
            
            )
            
            fig.write_image("Clusters "+ed_titles[a]+" "+titles[q]+".png", scale=2)
            




    
    ed_value_arrays=[[],[],[],[]]
    country_names=countries
    for j in range(len(countries)):
            country=countries[j]
            
            for a in range(len(ed_levels)):
                value=table[ed_levels[a]][country]
                if value==value:
                    ed_value_arrays[a].append(value)
    width = 0.15
    X_axis = np.arange(len(country_names))
    axs0[q].bar( X_axis , ed_value_arrays[0], width, label = 'Below upper secondary education')
    axs0[q].bar( X_axis +0.16, ed_value_arrays[1], width, label = 'Upper secondary and post-secondary non-tertiary education')
    axs0[q].bar(  X_axis+0.32, ed_value_arrays[2], width, label = 'Tertiary education')
    axs0[q].bar(  X_axis + 0.48, ed_value_arrays[3], width, label = 'All levels of education')
    axs0[q].set_xticks(X_axis+width, country_names)
    

    axs0[q].set_title("\n"+titles[q])

    axs0[q].set(xlabel='Countries', ylabel="Percentage (%)")

    axs0[q].legend(loc="lower right",ncols=4,prop={'size': 5})

    fig0.tight_layout()

fig0.savefig("social_connections"+".jpg", dpi=500)
fig0.show()
    