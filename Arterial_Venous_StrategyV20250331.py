# %% Import modules
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import dataprocessV20250331 as dp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr

# %% Arterial Venous Information
class Arterial_Venous_Strategy:
    def __init__(self, information: list, person: str, window: int, start: int, end: int):
        # person, window of single day
        self.information = information
        self.person = person
        self.window = window
        self.start = start
        self.end = end
        # read information of person
        self.all = dp.read_person(self.person)[self.information]
        self.day = sorted(list(set(self.all['透析日期'])))
        self.event = dp.event(self.person)
        try:
            self.event_date = list(set(self.event[self.event.columns[0]]))
        except:
            self.event_date = []
    
    def read_day(self, day: datetime.datetime):
        df_day = self.all[self.all['透析日期'] == day].set_index('治疗时间').drop('透析日期', axis=1).groupby('治疗时间').mean()
        data = {'动脉压': [np.nan] * 241, '静脉压': [np.nan] * 241}
        index = pd.Index([i for i in range(241)], name='治疗时间')
        df = pd.DataFrame(data, index=index)
        df.update(df_day)
        # for date, item in self.all.groupby('透析日期'):
        #     if date == day:
        #         item = abs(item[(item['治疗时间'] > self.start) & (item['治疗时间'] < self.end)].drop('透析日期', axis=1))
        #         item = item.groupby('治疗时间').mean()
        #         for i in range(118, 123):
        #             try:
        #                 item.loc[i, :] = item.loc[i-2: i+3, :].mean()
        #             except:
        #                 pass
        #         item.sort_index(ascending=True, inplace=True)
        df.sort_index(ascending=True, inplace=True)
        df = df.ffill().bfill()
        return df.loc[self.start: self.end,:]
        
    def read_day_mean(self, day: datetime.datetime):
        df = self.read_day(day=day)
        # calculate mean value
        mean_arterial = df.sort_values(by='动脉压', ascending=True, inplace=False).drop('静脉压', axis=1)[20: -20].mean().values[0]
        mean_venous = df.sort_values(by='静脉压', ascending=True, inplace=False).drop('动脉压', axis=1)[20: -20].mean().values[0]
        return pd.DataFrame({'动脉压': mean_arterial, '静脉压': mean_venous}, index=[day])
    
    def self_judge_strategy(self, threshold: float):
        Sample = []
        Sample_Dates = []
        for date in self.day:
            # df = self.read_day(day=date)
            df = self.read_day(day=date).rolling(window=window * 2, min_periods=1).mean()
            # for i in range(118, 123):
            #     df.loc[i, :] = df.loc[i-2: i+3, :].mean()
            Sample_Dates.append(date)
            Sample.append(df.T.values[1])
            break

        list_Disease_Judge = []
        list_Sum_Sample_Dates = []
        for date in tqdm(self.day):
            # df = self.read_day(day=date)
            df = self.read_day(day=date).rolling(window=window * 2, min_periods=1).mean()
            # for i in range(118, 123):
            #     df.loc[i, :] = df.loc[i-2: i+3, :].mean()
            # df.sort_index(ascending=True, inplace=True)
            array_test = df.T.values[1]
            list_Sum_Sample_Dates.append(len(Sample_Dates)/100)

            # Judge whether the array test is similar to the existing samples
            for sample in Sample:
                list_pearsonr = [0]
                # Pearsonr similarity
                list_pearsonr.append(abs(pearsonr(sample, array_test)[0]))
            if max(list_pearsonr) > threshold:
                list_Disease_Judge.append(1-max(list_pearsonr))
            # Not similar, maybe a disease date
            else:
                # Judge whether the date is in disease interval (60 days)
                for disease_date in self.event_date:
                    if disease_date - datetime.timedelta(days=30) <= date <= disease_date + datetime.timedelta(days=15):
                        list_Disease_Judge.append(1-max(list_pearsonr))
                        break
                # Not in disease interval, add to Sample list
                else:
                    Sample.append(array_test)
                    Sample_Dates.append(date)
                    list_Disease_Judge.append(1-max(list_pearsonr))
        # Plot
        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(10, 6))
        plt.scatter(self.day, list_Sum_Sample_Dates, color='green', marker='.', label='Cosum Samples')
        plt.scatter(self.event_date, [1] * len(self.event_date), color='orange', marker='o', label='Incident')
        plt.plot(self.day, pd.Series(list_Disease_Judge).rolling(window=15, min_periods=1).mean(), label='Probability')
        for i, x in enumerate(self.event_date):
            plt.axvline(x, linestyle='--', color='r', label='Disease Occur' if i == 0 else "")
            plt.axvspan(x-datetime.timedelta(days=self.window * 2), x+datetime.timedelta(days=self.window),
                        alpha=0.2, color='gray', label='Warning Period' if i == 0 else "")
        plt.title(f'Probability by using Arterial and Venous, Person {self.person}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_day(self, day: datetime.datetime):
        df = abs(self.read_day(day=day))
        plt.figure(figsize=(10, 6))
        plt.rcParams['figure.dpi'] = 300
        plt.plot(df.index, df['动脉压'], label='Arterial')
        plt.plot(df.index, df['静脉压'], label='Venous')
        plt.plot(df.index, df['动脉压'].rolling(window=self.window, min_periods=1).mean(), 
                label=f"Arterial Mean of {self.window} Minutes")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window, min_periods=1).mean(), 
                label=f"Venous Mean of {self.window} Minutes")
        plt.plot(df.index, df['动脉压'].rolling(window=self.window, min_periods=1).std(), 
                label=f"Arterial Std of {self.window} Minutes")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window, min_periods=1).std(),
                label=f"Venous Std of {self.window} Minutes")
        plt.title(f"Arterial and Venous Indicators of Patient {self.person} on Day {day}")
        plt.xticks(rotation=45)
        plt.xlabel('Minute')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_mean(self):
        df = abs(pd.concat([self.read_day_mean(day=day) for day in self.day], axis=0))
        plt.figure(figsize=(10, 6))
        plt.rcParams['figure.dpi'] = 300
        plt.plot(df.index, df['动脉压'], label='Arterial')
        plt.plot(df.index, df['静脉压'], label='Venous')
        plt.plot(df.index, df['动脉压'].rolling(window=self.window * 2, min_periods=1).mean(), 
                label=f"Arterial Mean of {self.window * 2} Days")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window * 2, min_periods=1).mean(), 
                label=f"Venous Mean of {self.window * 2} Days")
        plt.plot(df.index, df['动脉压'].rolling(window=self.window * 2, min_periods=1).std(), 
                label=f"Arterial Std of {self.window * 2} Days")
        plt.plot(df.index, df['静脉压'].rolling(window=self.window * 2, min_periods=1).std(),
                label=f"Venous Std of {self.window * 2} Days")
        for i, x in enumerate(self.event_date):
            plt.axvline(x, linestyle='--', color='r', label='Disease Occur' if i == 0 else "")
            plt.axvspan(x-datetime.timedelta(days=self.window * 2), x+datetime.timedelta(days=self.window),
                        alpha=0.2, color='gray', label='Warning Period' if i == 0 else "")
        plt.title(f"Arterial and Venous Indicators of Patient {self.person} for Long Period")
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

# %% Test '544 -'
information = ['透析日期', '治疗时间', '动脉压', '静脉压']
person = '544 -'
window = 5
start = 141
end = 200
person_544 = Arterial_Venous_Strategy(information, person, window, start, end)

print(person_544.all.head())
day = datetime.date(2023, 3, 9)
print(person_544.read_day_mean(day=day))
print(person_544.read_day(day=day).head())

print(person_544.plot_mean())
print(person_544.plot_day(day=day))
person_544.self_judge_strategy(threshold=0.5)

# %% Test '678 -'
information = ['透析日期', '治疗时间', '动脉压', '静脉压']
person = '678 -'
window = 5
start = 141
end = 200
person_678 = Arterial_Venous_Strategy(information, person, window, start, end)

print(person_678.all.head())
day = datetime.date(2023, 3, 9)
print(person_678.read_day_mean(day=day))
print(person_678.read_day(day=day).head())

print(person_678.plot_mean())
print(person_678.plot_day(day=day))
person_678.self_judge_strategy(threshold=0.5)

# %% Test '524 -'
information = ['透析日期', '治疗时间', '动脉压', '静脉压']
person = '524 -'
window = 5
start = 141
end = 200
person_524 = Arterial_Venous_Strategy(information, person, window, start, end)

print(person_524.all.head())
day = datetime.date(2023, 3, 9)
print(person_524.read_day_mean(day=day))
print(person_524.read_day(day=day).head())

print(person_524.plot_mean())
print(person_524.plot_day(day=day))
person_524.self_judge_strategy(threshold=0.5)

# %% Test '544 -'
df = pd.read_csv('bp.csv', encoding='GBK', index_col='id').loc['544 -']
df.sort_values(by='透析时间', ascending=True, inplace=True)
df.drop(['Unnamed: 0', '透析时间'], axis=1, inplace=True)
df = df.set_index('透析日期').groupby('透析日期').mean()
df.index = pd.to_datetime(df.index).date

# plot
plt.figure(figsize=(10, 6))
plt.rcParams['figure.dpi'] = 300
plt.plot(df.index, df['脉率'], label='ML')
plt.plot(df.index, df['收缩压'], label='SSY')
plt.plot(df.index, df['舒张压'], label='SZY')
plt.plot(df.index, df['脉率'].rolling(window=person_544.window * 2, min_periods=1).mean(), 
        label=f"ML Mean of {person_544.window * 2} Days")
plt.plot(df.index, df['收缩压'].rolling(window=person_544.window * 2, min_periods=1).mean(), 
        label=f"SSY Mean of {person_544.window * 2} Days")
plt.plot(df.index, df['舒张压'].rolling(window=person_544.window * 2, min_periods=1).mean(), 
        label=f"SZY Mean of {person_544.window * 2} Days")
plt.plot(df.index, df['脉率'].rolling(window=person_544.window * 2, min_periods=1).std(), 
        label=f"ML Std of {person_544.window * 2} Days")
plt.plot(df.index, df['收缩压'].rolling(window=person_544.window * 2, min_periods=1).std(), 
        label=f"SSY Std of {person_544.window * 2} Days")
plt.plot(df.index, df['舒张压'].rolling(window=person_544.window * 2, min_periods=1).std(),
        label=f"SZY Std of {person_544.window * 2} Days")
for i, x in enumerate(person_544.event_date):
    plt.axvline(x, linestyle='--', color='r', label='Disease Occur' if i == 0 else "")
    plt.axvspan(x-datetime.timedelta(days=person_544.window * 2), x+datetime.timedelta(days=person_544.window),
                alpha=0.2, color='gray', label='Warning Period' if i == 0 else "")
plt.title(f"BP Indicators of Patient {person_544.person} for Long Period")
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# %%
