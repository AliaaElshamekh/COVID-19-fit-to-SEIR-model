
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

plt.rcParams.update({'font.size': 14})
# Load data
# Latest available data on Feb 5 2020
data = pd.read_csv('2019_nCoV_20200121_20200206.csv', parse_dates = ['Last Update'])
data.shape

data[data['Province/State'] == 'Hubei'].head(8)

data = data.drop_duplicates()
data.shape

data['location'] = data['Country/Region'] + ' ' + data['Province/State'].fillna('')
ix = pd.date_range(start=data['Last Update'].min(), end=data['Last Update'].max(), freq='D')

daily = pd.DataFrame(columns=data.columns)

for item in data['location'].unique():
    _ = data[data['location']==item].set_index('Last Update')
    _ = _.resample('D').last().reindex(ix).fillna(method='ffill')
    _ = _.rename_axis('Date').reset_index()
    daily = daily.append(_, sort=False, ignore_index=True)

daily = daily.sort_values(['Date','Country/Region','Province/State'])
daily.tail()



def get_place(row):
    if row['Province/State'] == 'Hubei':
        return 'Hubei PRC'
    elif row['Country/Region'] == 'Mainland China':
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


death = pd.pivot_table(daily.dropna(subset=['Death']),
                         index='Date', columns='segment', values='Death', aggfunc=np.sum).fillna(method = 'ffill')
death.tail(10)


good = pd.pivot_table(daily.dropna(subset=['Recovered']),
                         index='Date', columns='segment', values='Recovered', aggfunc=np.sum).fillna(method = 'ffill')
good.tail(10)


plt.figure(figsize=(11,6))
plt.plot(good, marker='o')
plt.title('Recovered Cases')
plt.legend(good.columns)
plt.xticks(rotation=45)
plt.show()


current = daily.loc[daily['Date'] == np.max(daily['Date'])].groupby('segment').sum()
current['death rate'] = current['Death'] / current['Confirmed']
current['recovery rate'] = current['Recovered'] / current['Confirmed']
current[['death rate','recovery rate']] * 100

x=0