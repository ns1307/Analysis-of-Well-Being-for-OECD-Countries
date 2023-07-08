import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from numpy.linalg import norm
import plotly.figure_factory as ff


dataframe= pd.read_csv("employment_rate_by_edu.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]


queries=["BUPPSRY","UPPSRY_NTRY","TRY"]
ed_levels=["Below Upper Secondary","Upper Secondary non-Tertiary","Tertiary"]
years=list(np.arange(2006,2022))
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df
def empty_arr(arr):
    empty=True
    i=0
    while empty and i<len(arr):
        if arr[i]!=None and arr[i]==arr[i]:
            empty=False
        i=i+1
    return empty
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
        value=employment_table[year][country][index]
        values.append(value)
    return values
employment_table = pd.DataFrame(index=countries, columns=years)


for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = dataframe.loc[(dataframe['LOCATION']==countries[i])]#all data beloning to that country

        for year in years:
            year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==year)]
            bupps_df=year_df.loc[(year_df['SUBJECT']==queries[0])]
            upps_df=year_df.loc[(year_df['SUBJECT']==queries[1])]
            tri_df=year_df.loc[(year_df['SUBJECT']==queries[2])]

            if bupps_df.empty:
                bupps_value=None
            else:
                bupps_value=bupps_df.Value.to_numpy()[0]
            if upps_df.empty:
                upps_value=None
            else:
                upps_value=upps_df.Value.to_numpy()[0]
            if tri_df.empty:
                tri_value=None
            else:
                tri_value=tri_df.Value.to_numpy()[0]
            values=[bupps_value,upps_value,tri_value]
            
            employment_table[year][country]=values












for i in range(3):
    
    similarity_df=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    similarity_numbers=pd.DataFrame(index=countries, columns=countries,dtype=np.float64)
    drop_countries=[]
    for country in countries:
        values1=get_values_at_index(employment_table , country, i)
        if not empty_arr(values1):
            for other_country in countries:
                values2=get_values_at_index(employment_table , other_country, i)
                if not empty_arr(values2):
                    similarity, number_of_values = check_similarity(values1,values2)
                    similarity_df[country][other_country]=similarity
                    similarity_numbers[country][other_country]=number_of_values

        else:
            drop_countries.append(country)
    similarity_df=similarity_df.drop(drop_countries, axis=1)
    similarity_df=similarity_df.drop(drop_countries)
    for a in range(len(drop_countries)):
        cou=drop_countries[a]
        countries.remove(cou)
    import seaborn as sns 
    colormap = sns.color_palette("coolwarm", 50)
    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    sns.heatmap(similarity_df, annot=True, linewidths=.5, ax=ax,cmap=colormap,fmt=".3f")
    
    title="Similarity Matrix for "+ed_levels[i]
    plt.title(title)
    plt.xlabel("Countries")
    plt.ylabel("Countries")
    plt.savefig(title+".png", dpi=500)
    
    
    fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
    sns.heatmap(similarity_numbers, annot=True, linewidths=.5, ax=ax,cmap=colormap)
    
    
    title="Number of Comparisons for "+ed_levels[i]
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
    
    fig.write_image("Clusters "+ed_levels[i]+".png", scale=2)
    












for i in range(len(countries)):
    country=countries[i]
    bupps_values_emp=[]
    bupps_years_emp=[]
    upps_values_emp=[]
    upps_years_emp=[]
    tri_values_emp=[]
    tri_years_emp=[]
    
    for year in years:
        
        values_emp=employment_table[year][country]
        bupps_value_emp=values_emp[0]
        if bupps_value_emp!= None:
            bupps_values_emp.append(bupps_value_emp)
            bupps_years_emp.append(year)
        upps_value_emp=values_emp[1]
        if upps_value_emp!=None:
            upps_values_emp.append(upps_value_emp)
            upps_years_emp.append(year)
        tri_value_emp=values_emp[2]
        if tri_value_emp!=None:
            tri_values_emp.append(tri_value_emp)
            tri_years_emp.append(year)
    
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    best_degree=11
    legend=[]
    if len(bupps_values_emp)>0:
        plt.scatter(bupps_years_emp, bupps_values_emp,c="maroon")
        legend.append(ed_levels[0])
        mymodel = np.poly1d(np.polyfit(bupps_years_emp, bupps_values_emp, best_degree))
        myline = np.linspace(bupps_years_emp[0]-1,bupps_years_emp[-1]+2)
        plt.plot(myline, mymodel(myline), c="orangered")
        legend.append("Prediction for "+ ed_levels[0])
        text1="R2 score for "+ed_levels[0]+": "+str(round(r2_score(bupps_values_emp, mymodel(bupps_years_emp)),2))
        
    if len(upps_values_emp)>0:
        plt.scatter(upps_years_emp, upps_values_emp,c="navy")
        legend.append(ed_levels[1])
        mymodel = np.poly1d(np.polyfit(upps_years_emp, upps_values_emp, best_degree))
        myline = np.linspace(upps_years_emp[0]-1,upps_years_emp[-1]+2)
        plt.plot(myline, mymodel(myline), c="royalblue")
        legend.append("Prediction for "+ ed_levels[1])
        text2="R2 score for "+ed_levels[1]+": "+str(round(r2_score(upps_values_emp, mymodel(upps_years_emp)),2))
        
    if len(tri_values_emp)>0:
        plt.scatter(tri_years_emp , tri_values_emp,c="green")
        legend.append(ed_levels[2])
        mymodel = np.poly1d(np.polyfit(tri_years_emp, tri_values_emp, best_degree))
        myline = np.linspace(tri_years_emp[0]-1,tri_years_emp[-1]+2)
        plt.plot(myline, mymodel(myline), c="lime")
        legend.append("Prediction for "+ ed_levels[2])
        text3="R2 score for "+ed_levels[2]+": "+str(round(r2_score(tri_values_emp, mymodel(tri_years_emp)),2))
    text=text1+"\n"+text2+"\n"+text3
    plt.figtext(0,0, text, ha="left", fontsize=7)
    
    plt.legend(legend,prop={'size': 6})
   
    title="Employment Rates by Education Levels for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Employment Percentage by Edu Level (%)")
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)

    
            