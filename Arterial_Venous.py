# %%
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import dataprocess as dp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr

# %% Class Arterial_Venous
class Arterial_Venous:
    def __init__(self, information: list, person: str, window: int, start: int, end: int):
        # person, window of single day
        self.information = information
        self.person = person
        self.window = window
        self.start = start
        self.end = end
        # read information of person
        self.all = dp.read_person(self.person)[self.information]
    
    def read_day(self, day: str):
        for date, item in self.all.groupby('透析日期'):
            if date == day:
                item = abs(item[(item['治疗时间'] > self.start) & (item['治疗时间'] < self.end)].drop('透析日期', axis=1))
                item = item.groupby('治疗时间').mean()
                for i in range(118, 123):
                    try:
                        item.loc[i, :] = item.loc[i-2: i+3, :].mean()
                    except:
                        pass
                item.sort_index(ascending=True, inplace=True)
                return item
    
    def read_day_mean(self, day: str):
        for date, item in self.all.groupby('透析日期'):
            if date == day:
                item = abs(item[(item['治疗时间'] > self.start) & (item['治疗时间'] < self.end)].drop('透析日期', axis=1))
                item = item.groupby('治疗时间').mean()
                # calculate mean value
                mean_arterial = item.sort_values(by='动脉压', ascending=True, inplace=False).drop('静脉压', axis=1)[10: -10].mean()
                mean_venous = item.sort_values(by='静脉压', ascending=True, inplace=False).drop('动脉压', axis=1)[10:-10].mean()
                return pd.concat([mean_arterial, mean_venous], axis=0)

    def plot_day(self, df, *day):
        plt.figure(figsize=(10, 6))
        plt.rcParams['figure.dpi'] = 300
        plt.plot(df.index, df['动脉压'], label='Arterial')
        plt.plot(df.index, df['静脉压'], label='Venous')
        plt.plot(df.index, df['动脉压'].rolling(window=self.window, min_periods=1).mean(), 
                label=f"Arterial Mean of {window} Minutes")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window, min_periods=1).mean(), 
                label=f"Venous Mean of {window} Minutes")
        plt.plot(df.index, df['动脉压'].rolling(window=self.window, min_periods=1).std(), 
                label=f"Arterial Std of {window} Minutes")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window, min_periods=1).std(),
                label=f"Venous Std of {window} Minutes")
        plt.title(f"Arterial and Venous Indicators of Patient {self.person} on Day {day}")
        # plt.xticks(rotation=45)
        plt.xlabel('Minute')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    # def arterial_venous(self):


    #     return len(Sample)
    
    def mean_value_strategy(self, interval: int):
        list_arterial = []
        list_venous = []
        list_date = []
        for date, _ in self.all.groupby('透析日期'):
            arterial, venous = self.read_day_mean(date)
            list_arterial.append(arterial)
            list_venous.append(venous)
            list_date.append(date)
        df_arterial = pd.Series(list_arterial)
        df_venous = pd.Series(list_venous)
        # Generate mean value
        data = pd.concat([df_arterial, df_venous], axis=1)
        data.columns = ['动脉压', '静脉压']
        data.index = list_date
        # Plot
        return data


# %%
information = ['透析日期', '治疗时间', '动脉压', '静脉压']
person = '678 -'
window = 10
start = 20
end = 221
day = '2023-03-09'
df_event = dp.event(person)
list_event_date = list(set(df_event[df_event.columns[0]]))
df_person = Arterial_Venous(information, person, window, start, end)

# %%
df = df_person.mean_value_strategy(10)

interval = 10
plt.figure(figsize=(10, 6))
# plt.rcParams['figure.dpi'] = 300
plt.plot(pd.to_datetime(df.index), df['动脉压'], label='Arterial')
plt.plot(pd.to_datetime(df.index), df['静脉压'], label='Venous')
plt.plot(pd.to_datetime(df.index), df['动脉压'].rolling(window=interval, min_periods=1).mean(), 
        label=f"Arterial Mean of {interval * 3} Days")
plt.plot(pd.to_datetime(df.index), df['静脉压'].rolling(window=interval, min_periods=1).mean(), 
        label=f"Venous Mean of {interval * 3} Days")
for x in list_event_date:
    plt.axvline(x, linestyle='--', color='r')
    plt.axvspan(x-datetime.timedelta(days=30), x+datetime.timedelta(days=15), alpha=0.2, color='gray')

plt.title(f"Arterial and Venous Indicators of Patient {person} of Interval {interval * 3}")
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# %%
