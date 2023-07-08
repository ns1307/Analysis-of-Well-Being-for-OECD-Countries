import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("youth_unemployed.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]


queries=["MEN","WOMEN","AVG"]

years=list(np.arange(2006,2023))
unemployment_table = pd.DataFrame(index=countries, columns=years)
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
def get_values_at_index(table, country, index):
    values=[]
    for year in years:
        value=unemployment_table[year][country][index]
        values.append(value)
    return values


for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==countries[i])]#all data beloning to that country

        for year in years:
            year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==str(year))]
            men_df=year_df.loc[(year_df['SUBJECT']==queries[0])]
            women_df=year_df.loc[(year_df['SUBJECT']==queries[1])]
            
            if men_df.empty:
                men_value=None
            else:
                men_value=men_df.Value.to_numpy()[0]
            if women_df.empty:
                women_value=None
            else:
                women_value=women_df.Value.to_numpy()[0]
            if (men_value!=None and women_value != None):
                avg_value=(men_value+women_value)/2
            else:
                avg_value=None
            
            values=[men_value,women_value,avg_value]
            unemployment_table[year][country]=values







for i in range(3):

    similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    for country in countries:
        values1=get_values_at_index(unemployment_table , country, i)
        for other_country in countries:
            values2=get_values_at_index(unemployment_table , other_country, i)
            similarity, number_of_values = check_similarity(values1,values2)
            similarity_df[country][other_country]=similarity
            similarity_numbers[country][other_country]=number_of_values
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+queries[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+queries[i]
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
    for a in range (num_cluster):
        cluster=a
        cluster_name="Cluster "+str(cluster)
        array=[]
        for ind in range (len(countries)):
            if cluster_array[ind]==cluster:
                array.append(countries[ind])
        if len(array)!=0:
                df2 = {"cluster":cluster_name,'countries': array}
                cluster_table = cluster_table.append(df2, ignore_index = True)
        print("Cluster ",cluster,": ",array)
    
    
    
    fig = ff.create_table(cluster_table)
    
    fig.update_layout(
    autosize=True,
    
    )
    
    fig.write_image("Clusters "+queries[i]+".png", scale=2)
    


















for i in range(len(countries)):
    country=countries[i]
    avg_unemp_values=[]
    men_unemp_values=[]
    women_unemp_values=[]
    country_years=years.copy()
    for j in range (len(years)):
        year=years[j]
        unemp_values=unemployment_table[year][country]
        if unemp_values[2]==None:
            country_years.remove(year)
        else:
            avg_unemp_values.append(unemp_values[2])
            men_unemp_values.append(unemp_values[1])
            women_unemp_values.append(unemp_values[0])
        
    
    X=country_years
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
    print(country,"---degree ", best_degree,"--score: ",best_score)"""

    mymodel = np.poly1d(np.polyfit(X, avg_unemp_values , best_degree))
    myline = np.linspace(years[0]-1,years[-1]+2)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    plt.scatter(X, avg_unemp_values ,c="green")
    plt.scatter(X, men_unemp_values ,c="blue")
    plt.scatter(X, women_unemp_values ,c="red")
    text="R2 score:"+str(round(r2_score(avg_unemp_values, mymodel(X)),2))
    plt.figtext(0,0, text, ha="left", fontsize=10)
    
    plt.plot(myline, mymodel(myline), c="grey", linewidth=1)
    
    plt.legend(["Average Unemployed Youth","Unemployed Youth Men","Unemployed Youth Women",
                "Unemployment Rate Prediction for avg."])
    title="Unemploymed Youth Rate for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Unemployed Youth of Workable Youth Population (%)")
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)

    
            