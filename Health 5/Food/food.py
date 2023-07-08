import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff

charts=["FOODFRUI","FOODVEGG","FOODSUCR","FOODCALT","FOODTFAT","FOODPROT"]
chart_titles=[["Fruit Supply kilos per capita per year","kilos per capita (kg)"],
              ["Vegetables supply kilos per capita per year","kilos per capita (kg)"],
              ["Sugar supply kilos per capita per year","kilos per capita (kg)"],
              ["Total calories supply per capita per day","kilocalories per capita (kcal)"],
              ["Total fat supply grams per capita per day","grams per capita (gr)"],
              ["Total protein supply grams per capita per day","grams per capita (gr)"],
              ]
dataframe= pd.read_csv("food.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
colors=["steelblue","black","limegreen","crimson","royalblue","fuchsia",
        "darkgoldenrod","darkgreen","purple","orange","gold",
        "mediumorchid","darkred","mediumturquoise","grey","midnightblue",
        "black","black","black","black","black","black","black"]
cluster_colors=[["darkslategrey","black"],["red","black"],["blue","orange"],
                    ["pink","black"],
                    ["orange","black"],["green","grey"]]
perma_countries=countries.copy()
perma_colors=colors.copy()



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

for x in range(len(charts)):
    countries=perma_countries.copy()
    colors=perma_colors.copy()
    years=list(np.arange(2010,2022))
    table = pd.DataFrame(index=countries, columns=years)
    var_name=charts[x]
    filtered_df = dataframe.loc[(dataframe['VAR']==var_name)]
    
    for i in range(len(countries)):
         country=countries[i]
         country_df = filtered_df.loc[(filtered_df['COU']==country)]#all data beloning to that country
         for year in years:
             year_df=country_df.loc[(country_df['YEA']==year)]
             if year_df.empty:
                 percantage=None
             else:

                 percantage=year_df.Value.to_numpy()[0]
             table[year][country]=percantage
         
    
        
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
    
    title="Similarity Matrix"
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+" "+chart_titles[x][0]+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons"
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+" "+chart_titles[x][0]+".png", dpi=500)
    
    
    num_cluster=7
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

    fig.write_image("Clusters "+chart_titles[x][0]+".png", scale=2)

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
                        if value==value and value!=None:
                            values.append(value)
                    if len(values)!=0:
                        avg_value=sum(values)/len(values)
                        cluster_df[year][cluster_name]=avg_value
                table=table.append(cluster_df)
                countries.append(cluster_name)


    plt.figure(figsize=(10,10))
    c=0
    for i in range(len(countries)):
        
        country=countries[i]
        color=colors[i]
        
        values=table.loc[country].to_numpy().astype(float)
        
        years_filtered=[]
        values_filtered=[]
        for j in range(len(values)):
            if values[j]==values[j]:
                years_filtered.append(years[j])
                values_filtered.append(values[j])
        X=years_filtered
        y=values_filtered
        
        if("Cluster" in country):
            plt.plot(X, y,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[c][0] ,gapcolor=cluster_colors[c][1])
            c=c+1
        else:
            plt.plot( X, y,".-",label=country, c=color)
      
        """
        years=selected[i][0]
        percantages=selected[i][1]
        if(len(years)!=0):
            country=countries[i]
            
            j=0
            
            X=years
            y=percantages
            best_degree=6 #below comment line is tried, best result is with 6th degree
            best_degree=i
            best_score=0
            for i in range(5,10,1):
                mymodel = np.poly1d(np.polyfit(X, y, i))
                myline = np.linspace(country_years[0]-2,country_years[-1]+2)
                
                score=r2_score(y, mymodel(X))
                if score>best_score:
                    best_degree=i
                    best_score=score
                    
        
            mymodel = np.poly1d(np.polyfit(X, y, best_degree))
            myline = np.linspace(years[0]-1,years[-1]+1)
            
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
           
            plt.scatter(X, y,c="ORANGE")
            plt.legend(["actual values"])
            text="R2 score:"+str(round(r2_score(y, mymodel(X)),2))
            plt.figtext(0,0, text, ha="left", fontsize=10)
            
            plt.plot(myline, mymodel(myline))
            title="General Government Deficit Chart for "+ country
            plt.title(title)
            plt.xlabel("Year")
            plt.ylabel("Percantage of GDP (%)")
            plt.tight_layout()
            #dosya kayıt etmek için
            #plt.savefig(title+".png", dpi=1200)"""
    
    plt.xlabel("Year")
    plt.ylabel(chart_titles[x][1])
    plt.title(chart_titles[x][0])
    plt.legend(countries,prop={'size': 5})
    
    plt.tight_layout()
    
    plt.savefig(chart_titles[x][0]+".jpg", dpi=500)
    plt.show()
                