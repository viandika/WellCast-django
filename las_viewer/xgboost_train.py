import lasio
import numpy as np
import pandas as pd
import os
import json
from django.conf import settings


def load_data(filename):
    las_file = lasio.read(filename)
    data_well = las_file.df()
    log = list(data_well.columns.values)
    header = [{
        'WELL': filename,
        'START': las_file.well.STRT.value,
        'STOP': las_file.well.STOP.value,
        'STEP': las_file.well.STEP.value
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


def dataframing_train():
    '''
    TODO: fix directories and make it so it accepts uploaded las
    :return: data frame of train
    '''
    # Initialization
    train_source_dir = settings.BASE_DIR / 'las' / 'train'
    alias_file = settings.BASE_DIR / 'las' / 'alias.json'
    logs_selected = ['WELL', 'DEPTH', 'CAL', 'RXO', 'GR', 'POR', 'DRES', 'DT', 'DENS']

    with open(alias_file, 'r') as file:
        alias = json.load(file)

    # Loading raw data for train dataset
    data_train = pd.DataFrame()
    log_ava_train = pd.DataFrame()
    las_files = [file for file in train_source_dir.iterdir() if file.is_file()]
    for f in sorted(las_files):
        data_well, log_list = load_data(f)
        data_train = data_train.append(data_well)
        log_ava_train = log_ava_train.append(log_list)

    # Merge log aliases for train dataset
    data_train = data_train.reset_index()
    train = merge_alias(data_train, alias, logs_selected).dropna()
    train.rename(columns={'POR': 'NPHI', 'DENS': 'RHOB'}, inplace=True)

    # Select well data which has more than 5000ft length
    log_ava_train['LENGTH'] = log_ava_train['STOP'] - log_ava_train['START']
    log_ava_train = log_ava_train.sort_values('LENGTH', ascending=False)
    well_selected = log_ava_train[log_ava_train['LENGTH'] > 100]
    well_selected = well_selected['WELL']

    train = train[train['WELL'].isin(well_selected)]
    return train


def get_quartile(df, columns):
    quart_dict = {col: df[col].quantile([0.25, 0.75]).tolist() for col in columns}
    return quart_dict
