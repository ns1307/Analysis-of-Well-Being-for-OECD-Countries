import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


math_df= pd.read_csv("PISA_math.csv")
math_df=  math_df.loc[(math_df["SUBJECT"]=="TOT")]

read_df= pd.read_csv("PISA_read.csv")
read_df=  read_df.loc[(read_df["SUBJECT"]=="TOT")]

science_df= pd.read_csv("PISA_science.csv")
science_df=  science_df.loc[(science_df["SUBJECT"]=="TOT")]

countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
perma_countries= ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR']

colors=["steelblue","black","limegreen","crimson",
        "royalblue","fuchsia","darkgoldenrod","darkgreen",
        "purple","orange","gold","mediumorchid",
        "darkred","mediumturquoise","grey","midnightblue","black","black","black","black","black","black"]
perma_colors=colors.copy()
cluster_colors=[["darkslategrey","black"],["red","black"],["blue","black"],
                    ["pink","black"],
                    ["orange","black"],["green","grey"]]
    
years=list(np.arange(2006,2019,3))
table = pd.DataFrame(index=countries, columns=years)
selected=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

def calculate_avg(x1,x2,x3):
    if(x1!=None and x2!=None and x3 !=None):
        return (x1+x2+x3)/3
    else:
        if x1==None:#x1 null
            if x2==None:#x1 and x2 null
                if x3== None:#x1 and x2 and x3 null
                    return None
                else:#x1 and x2 null,  x3 not null
                    return x3
            else:#x1 null x2 not null
                if x3== None:#x1 and x3 null x2 not null
                    return x2
                else:#x1 null x2 and x3 not null
                    return (x3+x2)/2
        else:#x1 not null
            if x2==None:#x1 not null  x2 null
                if x3== None:#x1 not null, x2 and x3 null
                    return x1
                else:#x1 and x3 not null, x2 null
                    return (x3+x1)/2
            else:#x1 and x2 not null
                if x3== None:#x1 and x2 not null, x3 null
                    return (x1+x2)/2
                else:#x1 x2 x3 not null
                    return (x1+x2+x3)/2

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


for i in range(len(countries)):
    country=countries[i]
    math_values = math_df.loc[(math_df['LOCATION']==country)]#all data beloning to that country
    read_values= read_df.loc[(read_df['LOCATION']==country)]
    science_values=  science_df.loc[(science_df['LOCATION']==country)]

    for year in years:
        math=math_values.loc[(math_df['TIME']==year)]
        read=read_values.loc[(read_df['TIME']==year)]
        science=science_values.loc[(science_df['TIME']==year)]
        if math.empty:
            math=None
        else:
            math=math.Value.to_numpy()[0]
        if read.empty:
            read=None
        else:
            read=read.Value.to_numpy()[0]
        if science.empty:
            science=None
        else:
           science=science.Value.to_numpy()[0]
        avg=calculate_avg(math, read, science)
        values=[math,read,science,avg]
        
        table[year][country]=values
    






pisa_lessons=["MATH","READ","SCIENCE","AVG"]
for i in range(4):
    countries=perma_countries.copy()
    similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    for country in countries:
        values1=[]
        for year in years:
            value=table[year][country][i]
            values1.append(value)
        for other_country in countries:
            values2=[]
            for year in years:
                value=table[year][other_country][i]
                values2.append(value)
            
            similarity, number_of_values = check_similarity(values1,values2)
            similarity_df[country][other_country]=similarity
            similarity_numbers[country][other_country]=number_of_values
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+pisa_lessons[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+pisa_lessons[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    num_cluster=6
    from sklearn.cluster  import DBSCAN, OPTICS,AgglomerativeClustering,AffinityPropagation,SpectralClustering
    
    data_matrix = similarity_df
    model = AffinityPropagation(affinity='precomputed').fit(data_matrix)
    
    
    cluster_table = pd.DataFrame(columns=["cluster","countries"])
    print("Clusters for "+pisa_lessons[i])
    print(model.labels_)
    cluster_array=model.labels_
    clusters=[]
    for x in range (num_cluster):
        cluster=x
        cluster_name="Cluster "+str(cluster)
        array=[]
        for ind in range (len(countries)):
            if cluster_array[ind]==cluster:
                array.append(countries[ind])
        if len(array)!=0:
               df2 = {"cluster":cluster_name,'countries': array}
               cluster_table = cluster_table.append(df2, ignore_index = True)
        clusters.append(array)
        print("Cluster ",cluster,": ",array)
    print("-------------------------")
    

    fig = ff.create_table(cluster_table)

    fig.update_layout(
    autosize=True,

    )

    fig.write_image("Clusters "+pisa_lessons[i]+".png", scale=2)

    
    
    for x  in range (num_cluster):
        cluster=clusters[x]
        cluster_name="Cluster "+str(x)
        if(len(cluster)>0):
            cluster_countries=cluster
            cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
            for year in years:
                values=[]
                for country in cluster_countries:
                    value=table[year][country][i]
                    if value!=None :
                        values.append(value)
                if len(values)!=0:
                    avg_value=sum(values)/len(values)
                    if cluster_name in table.index:
                        arr=table[year][cluster_name]
                        
                    else:
                        arr=[]
                    arr.append(avg_value)
                    cluster_df[year][cluster_name]=arr
            if cluster_name in table.index:
                table = table.drop(cluster_name)
            table=table.append(cluster_df)
            countries.append(cluster_name)
        else:#if cluster has no value to append table
            cluster_df = pd.DataFrame(index=[cluster_name], columns=years)
            for year in years:
                if cluster_name in table.index:
                    arr=table[year][cluster_name]
                else:
                    arr=[]
                arr.append(None)
                cluster_df[year][cluster_name]=arr
            if cluster_name in table.index:
                table = table.drop(cluster_name)
            table=table.append(cluster_df)
            countries.append(cluster_name)
    a=0
    plt.figure(figsize=(5,8))
    title="Programme for International Student Assessment"+"\nPISA "
    for j in range(len(countries)):
        country=countries[j]
        color=colors[j]
        values=[]
        years_filtered=[]
        for q  in range (len(years)):
            year=years[q]
            value=table[year][country][i]
            if  value is not None:
                values.append(value)
                years_filtered.append(year)
        
        if len(values)>0:
            
            if("Cluster" in country):
                plt.plot(years_filtered,values,"--x",label=country,lw=0.5,markersize=3,c=cluster_colors[a][0] ,gapcolor=cluster_colors[a][1])
                a=a+1
            else:
                
                plt.plot( years_filtered,values,".-",label=country+" "+pisa_lessons[i],c=color)
                
    
    plt.title(title+pisa_lessons[i]+" RESULTS")
    
            
    plt.xlabel('Year')
    plt.ylabel("SCORE")
    
    plt.legend(countries,prop={'size': 5})


    plt.tight_layout()
        
    plt.savefig("PISA_"+pisa_lessons[i]+".jpg", dpi=500)
    plt.show()
        