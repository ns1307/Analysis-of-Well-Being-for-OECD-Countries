import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm


dataframe= pd.read_csv("purchase_power_parity.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]
currencies=[]

period_avg_df=dataframe.loc[(dataframe['TRANSACT']=="EXC")]
end_period_df=dataframe.loc[(dataframe['TRANSACT']=="EXCE")]

years=list(np.arange(2008,2023))
table = pd.DataFrame(index=countries, columns=years)

for i in range(len(countries)):
    country=countries[i]
    period_avg_country_df = period_avg_df.loc[(period_avg_df['LOCATION']==country)]
    end_period_country_df = end_period_df.loc[(end_period_df['LOCATION']==country)]
    currency=end_period_country_df.Unit.to_numpy()[0]
    currencies.append(currency)
    years_len=len(years)
    for i in range(years_len):
        values = []
        if i==0 :#if its first value
            avg_df=period_avg_country_df.loc[(period_avg_country_df['TIME']==years[i])]
            avg=avg_df.Value.to_numpy()[0]
            start=0
            end=0
        else:
            avg_df=period_avg_country_df.loc[(period_avg_country_df['TIME']==years[i])]
            avg=round(avg_df.Value.to_numpy()[0],2)
            start_df=end_period_country_df.loc[(end_period_country_df['TIME']==years[i-1])]
            start=round(start_df.Value.to_numpy()[0],2)
            end_df=end_period_country_df.loc[(end_period_country_df['TIME']==years[i])]
            end=round(end_df.Value.to_numpy()[0],2)
        year=years[i]
        values.append(avg)
        values.append(start)
        values.append(end)
        table[year][country]=values

def check_similarity(array1,array2):
    number_of_values=0
    new_array1=[]
    new_array2=[]
    for ind in range(len(array1)):
        value1=array1[ind]
        value2=array2[ind]
        if value1==value1 and value2==value2:#if values are not empty
            new_array1.append(value1)
            new_array2.append(value2)
            number_of_values=number_of_values+1
    cosine = np.dot(new_array1,new_array2)/(norm(new_array1)*norm(new_array2))
    #cosine=spatial.distance.cosine(new_array1, new_array2)
    return cosine,number_of_values
similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
for country in countries:
    
    avg_values1=[]
    for year in years:
        values=table[year][country]
        avg_values1.append(values[0])
    for other_country in countries:
        avg_values2=[]
        for year in years:
            values=table[year][other_country]
            avg_values2.append(values[0])
        similarity, number_of_values = check_similarity(avg_values1,avg_values2)
        similarity_df[country][other_country]=similarity
        similarity_numbers[country][other_country]=number_of_values
import seaborn as sns 
colormap = sns.color_palette("coolwarm", 50)
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(similarity_df, annot=True, linewidths=0.5, ax=ax,cmap=colormap,fmt=".3f")

title="Similarity Matrix Room per individual"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)


title="Number of Comparisons Room per individual"
plt.title(title)
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(title+".png", dpi=500)


num_cluster=6
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
    years=years
    country=countries[i]
    avg_values=[]
    start_values=[]
    end_values=[]
    for year in years:
        values=table[year][country]
        avg_values.append(values[0])
        start_values.append(values[1])
        end_values.append(values[2])
    j=0
    del end_values[0]
    del start_values[0]
    end_start_years=years.copy()
    del end_start_years[0]
    X=years
    y=avg_values
    """best_degree=6 #below comment line is tried, best result is with 6th degree
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
    myline = np.linspace(years[0]-1,years[-1]+1)"""
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    
    
    for x in range(len(end_start_years)):
        length=end_values[x]-start_values[x]
        if length<0:
            color="green"
        else:
            color="red"
        plt.bar(end_start_years[x], length,  bottom=start_values[x],color=color)
        
       

    plt.plot(years, avg_values,c="ORANGE",marker ='o')
    leg_dec = mpatches.Patch(color='Red', label='Decrease in currency (lost value against U.S. dollar) \nfrom start of the year to the end of the year')
    leg_inc = mpatches.Patch(color='green', label='Increase in currency (strengthened against U.S. dollar) \nfrom start of the year to the end of the year' )
    leg_avg= mpatches.Patch(color='orange', label='Yearly average')
    plt.legend(handles=[leg_avg,leg_inc,leg_dec],fontsize="6.5")
    title=currencies[i]+" Exchange Rate against U.S. Dollar ("+ country+")"
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("The equivalent of 1 U.S. Dollar in "+currencies[i])
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)

    
            