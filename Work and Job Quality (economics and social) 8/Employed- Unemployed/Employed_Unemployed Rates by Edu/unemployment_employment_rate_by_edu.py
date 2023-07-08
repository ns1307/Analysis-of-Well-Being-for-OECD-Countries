import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


dataframe= pd.read_csv("unemployment_rate_by_edu.csv")
countries = ['AUS','BEL','CAN','DNK','FIN','FRA','DEU','ITA','JPN','NLD','ESP','SWE','CHE','TUR','USA','GBR',]


queries=["BUPPSRY","UPPSRY_NTRY","TRY"]
ed_levels=["Below Upper Secondary","Upper Secondary non-Tertiary","Tertiary"]
years=list(np.arange(2006,2022))
unemployment_table = pd.DataFrame(index=countries, columns=years)
def getFromFiltered(df, column,row):
    new_df=df.loc[(df[column]==row)]
    return new_df



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
            
            unemployment_table[year][country]=values

dataframe= pd.read_csv("employment_rate_by_edu.csv")
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

for i in range(len(countries)):
    country=countries[i]
    bupps_values_emp=[]
    bupps_years_emp=[]
    upps_values_emp=[]
    upps_years_emp=[]
    tri_values_emp=[]
    tri_years_emp=[]
    bupps_values_unemp=[]
    bupps_years_unemp=[]
    upps_values_unemp=[]
    upps_years_unemp=[]
    tri_values_unemp=[]
    tri_years_unemp=[]
    for year in years:
        values_unemp=unemployment_table[year][country]
        bupps_value_unemp=values_unemp[0]
        if bupps_value_unemp!= None:
            bupps_values_unemp.append(bupps_value_unemp)
            bupps_years_unemp.append(year)
        upps_value_unemp=values_unemp[1]
        if upps_value_unemp!=None:
            upps_values_unemp.append(upps_value_unemp)
            upps_years_unemp.append(year)
        tri_value_unemp=values_unemp[2]
        if tri_value_unemp!=None:
            tri_values_unemp.append(tri_value_unemp)
            tri_years_unemp.append(year)
        
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
    
    
    
  
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    legend=[]
    if len(bupps_values_emp)>0:
        plt.scatter(bupps_years_emp, bupps_values_emp,c="GREEN")
        legend.append("Employed "+ ed_levels[0])
        
    if len(upps_values_emp)>0:
        plt.scatter(upps_years_emp, upps_values_emp,c="BLACK")
        legend.append("Employed "+ ed_levels[1])
        
    if len(tri_values_emp)>0:
        plt.scatter(tri_years_emp , tri_values_emp,c="ORANGE")
        legend.append("Employed "+ ed_levels[2])
        
        
        
        
    if len(bupps_values_unemp)>0:
        plt.scatter(bupps_years_unemp, bupps_values_unemp,c="YELLOW")
        legend.append("Unemployed "+ ed_levels[0])
        
    if len(upps_values_unemp)>0:
        plt.scatter(upps_years_emp, upps_values_unemp,c="RED")
        legend.append("Unemployed "+ed_levels[1])

    if len(tri_values_unemp)>0:
        plt.scatter(tri_years_unemp , tri_values_unemp,c="GREY")
        legend.append("Unemployed "+ed_levels[2])

    
    plt.legend(legend)
   
    title="Employment-Unemployment Rates by Education Levels for "+ country
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Percentage by Edu Level (%)")
    plt.tight_layout()
    #dosya kayıt etmek için
    plt.savefig(title+".png", dpi=500)

    
            