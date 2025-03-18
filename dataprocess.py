import pandas as pd


def event(number: str):
    df_event = pd.read_csv('event.csv', encoding='GBK')
    df_event[df_event.columns[2]] = pd.to_datetime(df_event[df_event.columns[2]], format='mixed').dt.date
    df_event.sort_values(by=[df_event.columns[1], df_event.columns[2]], ascending=True, inplace=True)
    df_event.drop(df_event.columns[0], axis=1, inplace=True)
    df_event.set_index('id', inplace=True)
    if number == 'all':
        return df_event
    else:
        try:
            return df_event.loc[number]
        except:
            return 'Patient is healthy'


def dialysis_process(number):
    df_dialysis_process = pd.read_csv('dialysis_process.csv', encoding='GBK')
    df_dialysis_process.sort_values(by=[df_dialysis_process.columns[1], df_dialysis_process.columns[-1]], ascending=True, inplace=True)
    df_dialysis_process.drop(df_dialysis_process.columns[0], axis=1, inplace=True)
    df_dialysis_process.set_index('id', inplace=True)
    if number == 'all':
        return df_dialysis_process
    else:
        try:
            return df_dialysis_process.loc[number]
        except:
            return 'No such patient'
    

def read_event(disease: str):
    df_event = pd.read_csv('event.csv', encoding='GBK')
    df_event[df_event.columns[2]] = pd.to_datetime(df_event[df_event.columns[2]], format='mixed').dt.date
    df_event.sort_values(by=[df_event.columns[1], df_event.columns[2]], ascending=True, inplace=True)
    df_event.drop(df_event.columns[0], axis=1, inplace=True)
    df_event.set_index('id', inplace=True)

    if disease == 'all':
        return df_event
    else:
        try:
            return df_event.loc[df_event[df_event.columns[1]] == disease]
        except:
            return 'No such disease'


def read_person(person: str):
    try:
        return pd.read_excel(f"{person}.xlsx", index_col='id')
    except:
        df_dialysis_process = pd.read_csv('dialysis_process.csv', encoding='GBK')
        try:
            df = df_dialysis_process.loc[df_dialysis_process[df_dialysis_process.columns[1]] == person]
        except:
            return 'No such patient'
        del df_dialysis_process
        df.sort_values(by=[df.columns[1], df.columns[-1]], ascending=True, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        df.set_index('id', inplace=True)
        df.to_excel(f"{person}.xlsx", index=True)
        return df