import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm


dataframe= pd.read_csv("long_term_unemployed.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]



years=list(np.arange(2000,2022))
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df
def check_similarity(array1,array2):
    number_of_values=0
    new_array1=[]
    new_array2=[]
    for ind in range(len(array1)):
        value1=array1[ind]
        value2=array2[ind]
        if value1==value1 and value2==value2 and value1!=None and value2!=None:#if values are not empty
            new_array1.append(value1)
            new_array2.append(value2)
            number_of_values=number_of_values+1
    cosine = np.dot(new_array1,new_array2)/(norm(new_array1)*norm(new_array2))
    #cosine=spatial.distance.cosine(new_array1, new_array2)
    return cosine,number_of_values
def get_values_at_index(table, country):
    values=[]
    for year in years:
        value=unemployment_table[year][country]
        values.append(value)
    return values
unemployment_table = pd.DataFrame(index=countries, columns=years)


for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==country)]#all data beloning to that country
        if not countr_filtered_df.empty:

            for year in years:
                year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==year)]
                
                tot_df=year_df.loc[(year_df['SUBJECT']=="TOT")]

                
                if tot_df.empty:
                    tot_value=None
                else:
                    tot_value=tot_df.Value.to_numpy()[0]
                
                unemployment_table[year][country]=tot_value

similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
for country in countries:
     values1=get_values_at_index(unemployment_table , country)
     for other_country in countries:
         values2=get_values_at_index(unemployment_table , other_country)
         similarity, number_of_values = check_similarity(values1,values2)
         similarity_df[country][other_country]=similarity
         similarity_numbers[country][other_country]=number_of_values
import seaborn as sns 
colormap = sns.color_palette("coolwarm", 50)
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(similarity_df, annot=True, linewidths=.5, ax=ax,cmap=colormap,fmt=".3f")
 
title="Similarity Matrix for Long Term Unemployed"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)
 
 
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
 
 
title="Number of Comparisons for Long Term Unemployed "
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)
 
 
num_cluster=8
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
    years_filtered=[]
    values=[]
    for year in years:
            value=unemployment_table[year][country]
            if value!=None and value==value:
                values.append(value)
                years_filtered.append(year)
    if len(years_filtered)>0:
        
    
        X=years_filtered
        y=values
        best_degree=17
        
        """
        best_score=0
        for i in range(4,18,1):
            degree=i
            mymodel = np.poly1d(np.polyfit(X, perc_values , degree))
            myline = np.linspace(years[0]-2,years[-1]+2)
            
            score=r2_score(perc_values, mymodel(X))
            if score>best_score:
                best_degree=i
                best_score=score
        print(country,"---degree ", best_degree,"--score: ",best_score)"""
    
        mymodel = np.poly1d(np.polyfit(X, y , best_degree))
        myline = np.linspace(X[0]-1,X[-1]+2)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
       
        plt.scatter(X, y ,c="green")
        text="R2 score:"+str(round(r2_score(y, mymodel(X)),2))
        plt.figtext(0,0, text, ha="left", fontsize=10)
        
        plt.plot(myline, mymodel(myline), c=color)
        
        plt.legend(["Lon Term Unemployed","Long Term Unemployed Prediciton"])
        title="Long Term Unemployment Rate for "+ country
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Percentage of Long Term Unemployed \nPeople in Unemployed People (%)")
        plt.tight_layout()
        #dosya kayıt etmek için
        plt.savefig(title+".png", dpi=500)

    
            