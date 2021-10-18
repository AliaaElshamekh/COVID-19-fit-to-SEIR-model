# Importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 14})
# Load data
# Latest available data on Feb 17 2020
# data = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv', parse_dates = ['Last Update'])
data = pd.read_csv('2019_nCoV_data.csv', parse_dates = ['Date','Last Update'])
data.shape

data[data['Province/State'] == 'Hubei'].tail(8)

# Clean up data on the first day (2020-01-22)

def clean_country(row):
    if row['Province/State'] == 'Hong Kong' or row['Province/State'] == 'Macau' or row['Province/State'] == 'Taiwan':
        return row['Province/State']
    elif row['Country'] == 'China':
        return 'Mainland China'
    else: return row['Country']

data['Country_adjust'] = data.apply(lambda row: clean_country(row), axis=1)

# Remove unnecessary columns
data = data.drop_duplicates()
data = data.drop(['Sno', 'Country', 'Last Update'], axis=1)
data = data.rename(columns={'Country_adjust': 'Country'})
data.head()

data['location'] = data['Country'] + ' ' + data['Province/State'].fillna('')
ix = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D', normalize=True)

daily = pd.DataFrame(columns=data.columns)

for item in data['location'].unique():
    _ = data[data['location']==item].set_index('Date')
    _ = _.resample('D').last().reindex(ix).fillna(method='ffill')
    _ = _.rename_axis('Date').reset_index()
    # _ = _.reset_index()
    daily = daily.append(_, sort=False, ignore_index=True)

daily = daily.sort_values(['Date','Country','Province/State'])
daily.tail()


def get_place(row):
    if row['Province/State'] == 'Hubei':
        return 'Hubei PRC'
    elif row['Country'] == 'Mainland China':
        return 'Others PRC'
    else:
        return 'World'


daily['segment'] = daily.apply(lambda row: get_place(row), axis=1)
daily.head()

confirm = pd.pivot_table(daily.dropna(subset=['Confirmed']), index='Date',
                         columns='segment', values='Confirmed', aggfunc=np.sum).fillna(method='ffill')
confirm.tail(10)

plt.figure(figsize=(11,6))
plt.plot(confirm, marker='o')
plt.title('Confirmed Cases')
plt.legend(confirm.columns)
plt.xticks(rotation=75)
plt.show()

death = pd.pivot_table(daily.dropna(subset=['Deaths']),
                         index='Date', columns='segment', values='Deaths', aggfunc=np.sum).fillna(method = 'ffill')
death.tail(10)

plt.figure(figsize=(11,6))
plt.plot(death, marker='o')
plt.title('Death Cases')
plt.legend(death.columns)
plt.xticks(rotation=45)
plt.show()

good = pd.pivot_table(daily.dropna(subset=['Recovered']),
                         index='Date', columns='segment', values='Recovered', aggfunc=np.sum).fillna(method = 'ffill')
good.tail(10)


plt.figure(figsize=(11,6))
plt.plot(good, marker='o')
plt.title('Recovered Cases')
plt.legend(good.columns)
plt.xticks(rotation=45)
plt.show()

df = confirm.join(death, lsuffix='_confirm', rsuffix='_death')
df = df.join(good.add_suffix('_recover'))
df['Hubei PRC_death_rate'] = df['Hubei PRC_death']/df['Hubei PRC_confirm']
df['Others PRC_death_rate'] = df['Others PRC_death']/df['Others PRC_confirm']
df['World_death_rate'] = df['World_death']/df['World_confirm']
df['Hubei PRC_recover_rate'] = df['Hubei PRC_recover']/df['Hubei PRC_confirm']
df['Others PRC_recover_rate'] = df['Others PRC_recover']/df['Others PRC_confirm']
df['World_recover_rate'] = df['World_recover']/df['World_confirm']

death_rate = df[['Hubei PRC_death_rate','Others PRC_death_rate','World_death_rate']]*100
plt.figure(figsize=(11,6))
plt.plot(death_rate, marker='o')
plt.title('Death Rate %')
plt.legend(death.columns)
plt.xticks(rotation=45)
plt.show()

recover_rate = df[['Hubei PRC_recover_rate','Others PRC_recover_rate','World_recover_rate']]*100
plt.figure(figsize=(11,6))
plt.plot(recover_rate, marker='o')
plt.title('Recovery Rate %')
plt.legend(good.columns)
plt.xticks(rotation=45)
plt.show()

df.iloc[-1, -6:]*100

x=0