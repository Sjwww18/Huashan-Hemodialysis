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

# %% Read Data
id = '544 -'
df_low_pressure = dp.read_event('低血压事件')
list_low_pressure_index = list(set(df_low_pressure.index))

df_person = dp.read_person(id)[['透析日期', '治疗时间', '动脉压', '静脉压']]
list_date = list(set(df_person['透析日期']))
len(list_date)

# %% Test
df_event = dp.event(id)
list_event_date = list(set(df_event[df_event.columns[0]]))
# print(list_event_date)

end = 221
start = 20
window = 30
person = id

# %%
def plot(plot_tuple):
    df, person, date, window = plot_tuple
    plt.figure(figsize=(10, 6))
    plt.rcParams['figure.dpi'] = 300
    plt.plot(df.index, df['动脉压'], label='Arterial')
    plt.plot(df.index, df['静脉压'], label='Venous')
    plt.plot(df.index, df['动脉压'].rolling(window=window, min_periods=1).mean(), 
             label=f"Arterial Mean of {window} Minutes")
    plt.plot(df.index, df['静脉压'].rolling(window=window, min_periods=1).mean(), 
             label=f"Venous Mean of {window} Minutes")
    plt.plot(df.index, df['动脉压'].rolling(window=window, min_periods=1).std(), 
             label=f"Arterial Std of {window} Minutes")
    plt.plot(df.index, df['静脉压'].rolling(window=window, min_periods=1).std(),
             label=f"Venous Std of {window} Minutes")
    plt.title(f"Arterial and Venous Indicators of Patient {person} on Day {date}")
    # plt.xticks(rotation=45)
    plt.xlabel('Minute')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()


for date, item in df_person.groupby('透析日期'):
    # if date < '2024-06-01':
    #     continue
    if date == '2023-03-09':
        item = abs(item[(item['治疗时间'] > start) & (item['治疗时间'] < end)].drop('透析日期', axis=1))
        item = item.groupby('治疗时间').mean()
        print(item.loc[115:125])
        for i in range(118, 123):
            try:
                item.loc[i, :] = item.loc[i-2: i+3, :].mean()
            except:
                pass
        plot_tuple = (item, person, date, window)
        plot(plot_tuple)
        print(len(item))
        break

# %% Train
Sample = []
Sample_Dates = []
for date, item in df_person.groupby('透析日期'):
    item = abs(item[(item['治疗时间'] > start) & (item['治疗时间'] < end)].drop('透析日期', axis=1))
    item = item.groupby('治疗时间').mean()
    # local mollification
    for i in range(118, 123):
        item.loc[i, :] = item.loc[i-2: i+3, :].mean()
    Sample_Dates.append(pd.to_datetime(date).date())
    Sample.append(item.T.values[0] + item.T.values[1])
    break

threshold = 0.8
list_Disease_Judge = []
for date, item in tqdm(df_person.groupby('透析日期')):
    item = abs(item[(item['治疗时间'] > start) & (item['治疗时间'] < end)].drop('透析日期', axis=1))
    item = item.groupby('治疗时间').mean()
    # local mollification
    try:
        for i in range(118, 123):
            item.at[i, :] = item.loc[i-2: i+3, :].mean()
    except:
        pass
    item.sort_index(inplace=True)
    array_test = item.T.values[0] + item.T.values[1]
    # Judge whether the array test is similar to the existing samples
    for sample in Sample:
        list_pearsonr = [0]
        # Pearsonr similarity
        try:
            list_pearsonr.append(abs(pearsonr(sample, array_test)[0]))
        except:
            pass
    if max(list_pearsonr) > threshold:
        list_Disease_Judge.append(1-max(list_pearsonr))
    # Not similar, maybe a disease date
    else:
        date = pd.to_datetime(date).date()
        # Judge whether the date is in disease interval (60 days)
        for disease_date in list_event_date:
            if disease_date - datetime.timedelta(days=30) <= date <= disease_date + datetime.timedelta(days=15):
                list_Disease_Judge.append(1-max(list_pearsonr))
                break
        # Not in disease interval, add to Sample list
        else:
            Sample.append(array_test)
            Sample_Dates.append(date)
            list_Disease_Judge.append(1-max(list_pearsonr))

# %% Plot

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 绘制折线图
plt.figure(figsize=(10, 6))
plt.scatter(list_event_date, [1] * len(list_event_date), color='orange', marker='o', label='Incident')
plt.plot(sorted(pd.to_datetime(list_date).date), pd.Series(list_Disease_Judge).rolling(window=15, min_periods=1).mean(), label='I')

for x in list_event_date:
    plt.axvline(x, linestyle='--', color='r')
    plt.axvspan(x-datetime.timedelta(days=30), x+datetime.timedelta(days=15), alpha=0.2, color='gray')

plt.title(f'Probability by using Arterial and Venous, Person {id}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend()
plt.show()
    

# %%

len(Sample)
# %% 
def I(Parameter_tuple):
    date, N = Parameter_tuple
    df_date = id_date.get_group(date)
    Arterial_Venous = df_date[['动脉压', '静脉压']].rolling(window=N)
    
    array_I = []
    for win in Arterial_Venous:
        # if len(win['静脉压']) == 1:
        #     # array_I.append(1)
        #     continue
        # else:
            # I = (win['静脉压'].std() + win['动脉压'].std()) * abs(pearsonr(win['动脉压'], win['静脉压'])[0]) / 20
        I = (win['静脉压'].std() + win['动脉压'].std()) / 20
        array_I.append(I)
    return pd.DataFrame(index=df_date['time'].values, data=array_I, columns=['I'])

# %% Test
date_N_tuple = [(date, 30) for date, _ in id_date]

df_list = []
for item in tqdm(date_N_tuple):
    df_list.append(I(item))
df = pd.concat(df_list)

# %%

df_low_pressure
# %%
import torch
import torch.nn as nn

# 定义一个简单的RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
input_size = 10
hidden_size = 20
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)

# 生成随机输入数据
batch_size = 32
seq_length = 5
x = torch.randn(batch_size, seq_length, input_size)

# 前向传播
output = model(x)
print(output.shape)

# %%
pearsonr([1, 2, 1, 2], [2, 1, 2, 0])[0]