import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


dataframe= pd.read_csv("employment_rate.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]


queries=["MEN","WOMEN","TOT"]

years=list(np.arange(2006,2023))
employment_table = pd.DataFrame(index=countries, columns=years)
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df


working_df = dataframe.loc[(dataframe['MEASURE']=="PC_WKGPOP")]#all data beloning to that country

for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = working_df.loc[(working_df['LOCATION']==countries[i])]#all data beloning to that country

        for year in years:
            year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==str(year))]
            men_df=year_df.loc[(year_df['SUBJECT']=="MEN")]
            women_df=year_df.loc[(year_df['SUBJECT']=="WOMEN")]
            tot_df=year_df.loc[(year_df['SUBJECT']=="TOT")]

            if men_df.empty:
                men_value=None
            else:
                men_value=men_df.Value.to_numpy()[0]
            if women_df.empty:
                women_value=None
            else:
                women_value=women_df.Value.to_numpy()[0]
            if tot_df.empty:
                tot_value=None
            else:
                tot_value=tot_df.Value.to_numpy()[0]
            values=[men_value,women_value,tot_value]
            
            employment_table[year][country]=values

dataframe= pd.read_csv("unemployment_rate.csv")

unemployment_table = pd.DataFrame(index=countries, columns=years)
working_df = dataframe.loc[(dataframe['MEASURE']=="PC_LF")]#all data beloning to that country

for i in range(len(countries)):
        country=countries[i]
        countr_filtered_df = working_df.loc[(working_df['LOCATION']==countries[i])]#all data beloning to that country

        for year in years:
            year_df=countr_filtered_df.loc[(countr_filtered_df['TIME']==str(year))]
            tot_df=year_df.loc[(year_df['SUBJECT']=="TOT")]

            
            if tot_df.empty:
                tot_value=None
            else:
                tot_value=tot_df.Value.to_numpy()[0]
            values=[tot_value]
            
            unemployment_table[year][country]=values

for i in range(len(countries)):
    country=countries[i]
    tot_values=[]
    men_values=[]
    women_values=[]
    tot_unemp_values=[]
    country_years=years.copy()
    for j in range (len(years)):
        year=years[j]
        values=employment_table[year][country]
        unemp_values=unemployment_table[year][country]
        men_values.append(values[0])
        women_values.append(values[1])
        tot_values.append(values[2])
        if not unemp_values[0]:
            del country_years[j]
        else:
            tot_unemp_values.append(unemp_values[0])
   
    X=years
    unemp_years=country_years
    best_degree=11
    """
    best_degree=i
    best_score=0
    for i in range(5,18,1):
        mymodel = np.poly1d(np.polyfit(X, tot_values, i))
        myline = np.linspace(years[0]-2,years[-1]+2)
        
        score=r2_score(tot_values, mymodel(X))
        if score>best_score:
            best_degree=i
            best_score=score
    print(country,"---degree ", best_degree,"--score: ",best_score)"""

    mymodel = np.poly1d(np.polyfit(X, tot_values, best_degree))
    myline = np.linspace(years[0]-1,years[-1]+2)
    
    unemp_model = np.poly1d(np.polyfit(unemp_years, tot_unemp_values, best_degree))
    unemp_myline = np.linspace(years[0]-1,years[-1]+2)
    
    fig = plt.figure(figsize=(8,9))
    ax = fig.add_subplot(111)
   
    plt.scatter(X, tot_values,c="GREEN")
    plt.scatter(X, women_values,c="RED")
    plt.scatter(X, men_values,c="BLUE")
    plt.scatter(unemp_years, tot_unemp_values,c="BLACK")
    text1="R2 score for emp:"+str(round(r2_score(tot_values, mymodel(X)),2))
    text2="R2 score for unemp:"+str(round(r2_score(tot_unemp_values, unemp_model(unemp_years)),2))
    plt.figtext(0,0, text1+"\n"+text2, ha="left", fontsize=10)
    plt.plot(myline, mymodel(myline), c="GREY")
    plt.plot(unemp_myline, unemp_model(unemp_myline), c="orange")
    plt.legend(["Total emplpyed","Employed women","Employed men","Total Unemployed"
                ,"Employment Rate Prediction","Unemployment Rate Prediction"])

    title="Employment and Unemployment Rate Chart for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Employment and Unemployment Rate of Workable Population (%)")
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)

    
            