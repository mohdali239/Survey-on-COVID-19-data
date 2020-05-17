# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 03:11:54 2020

@author: Muhammad Ali
"""

#importing all the  important libraries 

import numpy as np
import pandas as pd 
import matplotlib.pylot as plt 
import matplotlib.colors as mcolors
import random
import math
import time 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from skearn.svm import svm
from sklarn.metrics import mean_squared_error, mean_absolute_error
import datetime
import opertor
plt.style.use('seaborn')
%matplotlib inline

#Loading all  the three data sets
confirmed_cases = pd.read_csv('C:/Users/Muhammad Ali/Downloads/novel-corona-virus-2019-dataset')
deaths_reported = pd.read_csv('C:/Users/Muhammad Ali/Downloads/novel-corona-virus-2019-dataset')
recovered_cases = pd.read_csv('C:/Users/Muhammad Ali/Downloads/novel-corona-virus-2019-dataset')


#Display the head of the datasets
confirmed_cases.head()
deaths_reported.head()
recovered_cases.head()


#Extracting all the columns using the .keys() functon

cols = confirmed_ cases.keys()
cols


#Extracting all the colums that have information of confirmed ,deaths abd recovered cases

confirmed = confirmed_cases.loc[:,cols[4]:cols[-1]]
deaths = deaths_reported.loc[:,cols[4]:cols[-1]]
recoveries = recovered_cases.loc[:,cols[4]:cols[-1]]

#hech the head of the outbreak caess
confirmed.head()


#Finding the total confirmed cases, death cases and recoveerd cases and append  them to an 4 empty lists 
#Also calculate the matalisty rate whhich is the death_sum/confirmed cases

dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
total_recovered = []


for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths.sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confiremed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
    
#Let's display each of the newly created varible 

confirmed_sum
death_sum
recovered_sum
world_cases


#Convert all the dates and cases in the form of a numpy array

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1) 
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)

days_since_1_22
world_cases
total_deaths
total_recovered


#Future forecasting for the next 10 days
days_in_future = 10
future_forecast = np.array([i for i in range(len(dates)+days_future)]).reshape(-1, 1)
adjuted_dates = future_forecast[:-10]

future_forecast

#Convert all the integers into datetime for better visualization
start = '1/22/2020'
start_date = datetime.datetime.strptime(start,%m%d%Y)
future_forcast_date = []
for i in range(len(future_forcast)):
    future_forcast date append((start_date + datetime.timedelta(days=1)).strftime('%m%d%Y'))
    
  
#For visualization with the latest data of 15th of march

latest_confimed = confirmed_cases[dates[-1]]
latest_deaths = deaths_reported[dates[-1]]
latest_recoveries = recovered_cases[dates[-1]]

latest_confirmed
latest_deaths
latest_recoveries


#Find the List of unique countries
unique_countries = list(confirmed_cases['Country/Region'].unique())
unique_countries

#The next line of code will basically calculate the total number of confirmed cases by each country
country_confirmed_cases = []
no_cases = []
for i unique_countries:
    cases = latest_confired[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
        
for i in no_cases:
    unique_countries.remove(i)

unique_countries = [k for k, v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=0)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()


#number of cases per country/region
print('Confirmed Cases by Contries/Region:')
for i in range(len(unique_contries)):
    print(f'{unique_ountries[i0]}: {country_confirmed_cases[i]} cases')
    
    
#Find the list of unique provinces
 
unique_provinces = list(confirmed_cases['Provinces/states'].unique())

#those are contries which are not provines/states.

outliers = ['united kingdom','Denmark','France'] 
for i in outliers:
    unique_provinces.remove(i)
    
    
#Finding the number of confirmed cases per province,state or city

province_confirmed_cases = []
no_cases = []
for i in unique_provinces:
    cases = latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
        
for i in no_cases"
    unique_provinces.remove(i)

#number of cases per provvince/state/city

for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}: {province_confirmed_cases}')


#handling non values if there is any

nan_indices = []

for i in range(len(unique_province)):
    if type(unique_provices[i]) == float:
        nan_indices.append(i)
        
        
unique_provinces = list(unique_provinces)
province_confirmed_cases = list(province_confirmed_cases)

for i in nan_indices:
    unique_provices.pop(i)
    province_confirmed_cases.pop(i)     
    
    
# Plot a bar graph to see the total confirmed cases across different countries

plt.figure(figsize=(32,32))
plt.barh(unique_countries, country_confirmed_cases)
plt.title('Number of Covid-19 Confirmed Cases in Countries')
plt.xlabel('Number of Covid Confirmed Caese')
plt.show()

# Plot a bar graph to see the total confirmed cases b/w mainland china and outside mainland china

china_confirmed = latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()
outside_mainland_china_confirmed = np.sum(country_confirmed_cases)-china_confirmed
plt.figure(figsize=(16, 9))
plt.barh('Mainland China',china_confirmed)
plt.barh('Outside Mainland China',outside_mainland_china_confirmed)
plt.title('Number of Confirmed Coronavirus cases')
plt.show()

# Print the total cases in mainland china outside of it

print('Outside Mainland China{} cases:',format(outside_mainland_china_confirmed))
print('Mainland China: {} cases',format(china_confrmed))
print('Total:{} cases'.format(china_confirmed+outside_mainland_china_confirmed))

# Only show 10 countries with the most confirmed cases,the rest are grouped into the categary named others

visual_unique_countries = [] 
visual_confirmed_cases = []
other = np.sum(country_confirmed_cases[10:])
for i in range(len(contry_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
visual_unique_countries.append('Others')
visual_confirmed_cases.append(others) 


# Visualize the 10 countries
plt.figure(figsize=(32, 18))
plt.barh(visual_unique_countries, visual_confirmed_cases)
plt.title('Number of Covid-19 Confirmed Cases in Countries/Regions',size=20)
plt.show()


# Create a pie chart to see the total confirmed cases in 10 different countries
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len()unique_countries) 
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases per Country')
plt.pie(visual_confirmed_cases,colors=c)
plt.legend(visual_unique_countries,loc='best')
plt.show()


# Create a pie chart to see the total confirmed cases in 10 different countries outside china

c = random.choices(len(mcolors.CSS4_COLORS.values()), k = len(unique_countries)) 
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases in Countries Outside of Mainlend China')
plt.pie(visual_confirmed_cases[1:], colors=c)
plt.legend(visual_unique_countries[1:],loc = 'best')
plt.show()

# Building the SVM model

kernel = ['poly','sigmoid','rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel,'C': c,'gamma': gamma,'epsilon': epsilon, 'shrinking' : shrinking}
     
svm = SVM()
svm_search = RandomizedSearchCV(svm, svm_grid,scoring = 'neg_mean_squared_error',cv = 3,return_train_score=True,n_jobs=-1, n_iter=40,verbose=1)
svm_search.best_params_

svm_confirmed = svm_search.best_estimator_
svm_pred = svm_confirmed.predict(future_forecast)

svm_confirmed
svm_pred

# check against testing data

svm_test_pred = svm_confirmed.predict(x_text_confirmed)
plt.plot(svm_text_pred)
plt.plot(y_text_confirmed)
print('MAE:', mean_absolute_error(svm_text_pred, y_text_confirmed))
print('MSE:',mean_squared_error(svm_text_pred,y_text_confirmed))

# Total Number of coronavirus cases over time
 
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.title('Number of Conronavirus Cases Over Time',size=30)
plt.xlabel('Day Since 1/22/2020',size=30)
plt.ylabel('Number of Cases',size=30)
plt.xticks(size=15)
plt.yticks(size=15)


# Confirmed vs Predicted cases

plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forcast, svm_pred, linestyle='deshed',color='purple')
plt.title('Number of Coronavirus Cases over Time',size=30)
plt.xlabel('Days Since 1/22/2020',size=30)
plt.ylabel('Number of Cases',size=30)
plt.legend(['Confirmed Cases', 'svm predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# prediction for the next 10 days using svm 

print('svm future prediction:')
set(sip(future_forcast_dates[-10:],svm_pred[-10:]))


# usinglinear regression model to make predictions

from sklearn.linear_model import LinearRegression
linear_model = LinearRegrssion(normalize=True, fit_intercept=True)
linear_model.fit(X_train_Confirmed, y_train_confirmed)   
test_linear_pred = linear_model.predict(x_text_confirmed)
linear_pred = linear_model.predict(future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred,y_test_confirmed))

# gp
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)

    
plt.figure(figsize=(20, 12))
plt.plot(adjusted_date, world_cases)
plt.plot(future_forcast, linear_pred,linestyle='dashed',color='orange')
plt.title('Number of Coronavirus Cases Over Time',size=30)
plt.xlabel('Days Since 1/22/2020',size=30) 
plt.ylabel('Number of cases',size=30)
plt.legend(['Confirmed Cases', 'Linear Regression Prediction'])  
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# Prediction for the next 10 days using Linear Regression 

print('Linear regression future predictions:')
print(linear_pred[-10:])

# Total deaths over time 

plt.figure(figsze=(20, 12))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.title('Number of Coronavirus Deaths Over Time',size=30)
plt.xlabel('Time',size=30)
plt.ylabel('Number of Deaths',size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


mean_mortality_rate = np,mean(mortality_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate, linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time',size=30 )
plt.legend(['mortality rate',y='+str(mean_mortality_rate)])
plt.xlabel('Time', size=30)
plt.ylabel('Mortality Rate',size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# Coronavirus  Cases Recovered Over time

plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, mortality_rate, color='green')
plt.title('Number of Coronavirus Cases Recoverd Over Time',size=30)
plt.xlabel('Time',size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# Number of Coronavirus cases recovered vs the number of deaths  

plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['deaths',recoveries], loc='best',fontsize=20)
plt.title('Number of Coronavirus Cases', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Number of Cases',size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# Coronavirus Deaths vs Recoveries

plt.figure(figsize=(20, 12))
plt.plot(total_recovered, total_deaths)
plt.title('Coronavirus Deaths vs Coronavirus Recoveries', size=30)
plt.xlabel('Total number of Coronavirus Recoveries', size=30)
plt.ylabel('Total number of Coronavirus Deaths', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
                 