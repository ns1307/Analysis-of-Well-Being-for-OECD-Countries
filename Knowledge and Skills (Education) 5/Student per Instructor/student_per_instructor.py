import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff



dataframe= pd.read_csv("student_per_instructor.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
perma_countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]

colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]

perma_colors=colors.copy()

cluster_colors=[["orange","black"],["darkslategrey","black"],
                ["red","black"],["pink","black"],
                ["blue","black"],["green","grey"]]

missing = dataframe[ (dataframe['Flag Codes'] =="o") | (dataframe['Flag Codes'] =="k") | 
                    (dataframe['Flag Codes'] =="m") | (dataframe['Flag Codes'] =="w") ].index
dataframe.drop(missing , inplace=True)
filtered_df=dataframe.filter(countries)
filtered_df = dataframe.loc[(dataframe['INDICATOR']=="PERS_RATIO_INST")]
filtered_df = filtered_df.loc[(filtered_df['REF_SECTOR']=="INST_T")]
ed_levels=[["ISCED11_0","Early Childhood Education"],["ISCED11_01","Early Childhood Educational Development"],
           ["ISCED11_02","Pre-primary Education",],["ISCED11_1","Primary Education"],
           ["ISCED11_2","Lower Secondary Education"],["ISCED11_2_3","Secondary Education"],
           ["ISCED11_3","Upper Secondary Education"],["ISCED11_34","Upper Secondary General Education"],
           ["ISCED11_35","Upper Secondary Vocational Education"],["ISCED11_4","Post-secondary non-tertiary Education"],
           ["ISCED11_44","Post-secondary non-tertiary General Education"],
           ["ISCED11_45","Post-secondary non-tertiary Vocational Education"],
           ["ISCED11_5","Short-cycle Tertiary Education"], ["ISCED11_5T8","Tertiary Education"],
           ["ISCED11_6T8","Bachelor's, Master's and Doctoral or Equivalent Level"]
                ]

years=list(np.arange(2010,2021))




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
for q in range (len(ed_levels)):
    countries=perma_countries
    colors=perma_colors.copy()
    new_colors=[]
    new_countries=[]
    no_data_countries=[]
    table = pd.DataFrame(index=countries, columns=years)
    
    ed_level_df = filtered_df.loc[(filtered_df['EDUCATION_LEV']==ed_levels[q][0])]
    for i in range(len(countries)):
        country=countries[i]
        color=colors[i]
        country_df = ed_level_df.loc[(ed_level_df['COUNTRY']==country)]#all data beloning to that country
        no_value_interval = country_df[ (country_df['YEAR'] <2010)].index
        country_df.drop(no_value_interval , inplace=True)
        if not country_df.empty:
            for year in years:
                year_df=country_df.loc[(country_df['YEAR']==year)]
                if not year_df.empty:
                    percantage=year_df.Value.to_numpy()[0]
                    table[year][country]=percantage
            new_countries.append(country)
            new_colors.append(color)
        else:
            table = table.drop(country)
            no_data_countries.append(country)

    
    countries=new_countries
    colors=new_colors+["black","black","black","black","black","black"]

    

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
    
    title="Similarity Matrix for "+ed_levels[q][1]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+ed_levels[q][1]
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
    print("-------------")        
    fig = ff.create_table(cluster_out_table)

    fig.update_layout(autosize=True,)
    fig.write_image("Clusters "+ ed_levels[q][1]+".png", scale=2)
    
    for x  in range (num_cluster):
        cluster=clusters[x]
        cluster_name="Cluster "+str(x)
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
    
    
    a=0
    plt.figure(figsize=(5,5))
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
       
        
        if("Cluster" in country):
            plt.plot(X,y,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[a][0] ,gapcolor=cluster_colors[a][1])
            a=a+1
        else:
            plt.plot(X,y,".-",lw=1.5,markersize=12,label=country,c=color)
        
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
    plt.ylabel("Ratio")
    plt.title("Students per Instructor in \n"+ed_levels[q][1])
    plt.legend(countries,prop={'size': 5})
    
    text="No data for: "+', '.join(no_data_countries)
    plt.figtext(0,0, text, ha="left",fontweight="bold", fontsize=7)
    
    plt.tight_layout()
    
    plt.savefig("Students per Instructor in "+ed_levels[q][1]+".jpg", dpi=500)
    plt.show()
                