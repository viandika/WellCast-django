import lasio
import numpy as np
import pandas as pd
import os
import json


def load_data(filename):
    las_file = lasio.read(filename)
    data_well = las_file.df()
    log = list(data_well.columns.values)
    header = [{
        'WELL': filename,
        'START': las_file.well.STRT.value,
        'STOP': las_file.well.STOP.value,
        'STEP': las_file.well.STEP.value,
        'LAT': las_file.well.SLAT.value,
        'LON': las_file.well.SLON.value,
        'DATUM': las_file.well.DATUM.value
    }]
    data_well['WELL'] = filename

    log_list = pd.DataFrame(header)
    for newlog in log:
        log_list[newlog] = "v"
    return data_well, log_list


def merge_alias(db, alias, logs_selected):
    well = db['WELL'].unique()
    merged_data = pd.DataFrame()

    for i in range(len(well)):
        data = db.where(db['WELL'] == well[i]).dropna(axis=1, how='all')
        for j in range(len(alias)):
            welllog_name = list(set(data.columns).intersection(alias.get(list(alias)[j])))
            samelog = data[welllog_name]
            count_log = dict(sorted(zip(welllog_name, samelog.count()), key=lambda item: item[1], reverse=True))
            welllog_name = list(count_log.keys())
            if len(welllog_name) != 0:
                # If more than one log aliases exist, normalize each log to have same data range in the same depth
                if len(welllog_name) > 1:
                    alias_logs = data[welllog_name].dropna()
                    if (list(alias)[j] not in ['CAL', 'DTCO', 'DTSM']) and (len(alias_logs) != 0):
                        a = []
                        b = []
                        c = []
                        for n in range(len(alias_logs.columns)):
                            q1 = alias_logs[welllog_name[n]].quantile(0.1)
                            q9 = alias_logs[welllog_name[n]].quantile(0.9)
                            a.append(q1)
                            b.append(q9)
                            c = [b - a for (a, b) in zip(a, b)]
                            c = list(map(lambda x: x / c[0], c))
                        for n in range(len(welllog_name)):
                            data.loc[:, welllog_name[n]] *= 1 / c[n]
                    for k in range(len(welllog_name) - 1):
                        data[welllog_name[0]].fillna(data[welllog_name[k + 1]], inplace=True)
                data[list(alias)[j]] = data[welllog_name[0]]
        merged_data = merged_data.append(data)
        merged_data = merged_data[merged_data['WELL'].notna()]
    merged_data = merged_data[logs_selected]
    return merged_data


# Initialization
current_dir = os.getcwd()
train_source_dir = '/data/train'
test_source_dir = '/data/test'
alias_file = '/data/alias.json'
logs_selected = ['WELL', 'DEPT', 'CAL', 'SP', 'GR', 'POR', 'DRES', 'DTCO', 'PEF', 'DENS', 'DRHO']

with open(current_dir + alias_file, 'r') as file:
    alias = json.load(file)

# Loading raw data for train dataset
data_train = pd.DataFrame()
log_ava_train = pd.DataFrame()
for f in sorted(os.listdir(current_dir + train_source_dir)):
    data_well, log_list = load_data(current_dir + f"/{train_source_dir}/{f}")
    data_train = data_train.append(data_well)
    log_ava_train = log_ava_train.append(log_list)

# Merge log aliases for train dataset
data_train = data_train.reset_index()
train = merge_alias(data_train, alias, logs_selected).dropna()
train.rename(columns={'POR':'NPHI', 'DENS':'RHOB'}, inplace=True)

# Select well data which has more than 5000ft length
log_ava_train['LENGTH'] = log_ava_train['STOP'] - log_ava_train['START']
log_ava_train = log_ava_train.sort_values('LENGTH', ascending=False)
well_selected = log_ava_train[log_ava_train['LENGTH'] > 10000]
well_selected = well_selected['WELL']

train = train[train['WELL'].isin(well_selected)]

# Loading raw data for test dataset
data_test = pd.DataFrame()
log_ava_test = pd.DataFrame()
for f in sorted(os.listdir(current_dir + test_source_dir)):
    data_well, log_list = load_data(current_dir + f"/{test_source_dir}/{f}")
    data_test = data_test.append(data_well)
    log_ava_test = log_ava_test.append(log_list)

# Merge log aliases for test dataset
data_test = data_test.reset_index()
test = merge_alias(data_test, alias, logs_selected)
test.rename(columns={'POR':'NPHI', 'DENS':'RHOB'}, inplace=True)

# Save loaded data
train.to_csv('data/preprocessed/train.csv')
test.to_csv('data/preprocessed/test.csv')
