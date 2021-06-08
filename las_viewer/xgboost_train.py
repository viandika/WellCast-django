import lasio
import numpy as np
import pandas as pd
import os
import json
from django.conf import settings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def load_data(filename):
    las_file = lasio.read(filename)
    data_well = las_file.df()
    log = list(data_well.columns.values)
    header = [
        {
            "WELL": filename,
            "START": las_file.well.STRT.value,
            "STOP": las_file.well.STOP.value,
            "STEP": las_file.well.STEP.value,
        }
    ]
    data_well["WELL"] = filename

    log_list = pd.DataFrame(header)
    for newlog in log:
        log_list[newlog] = "v"
    return data_well, log_list


def merge_alias(db, alias, logs_selected):
    well = db["WELL"].unique()
    merged_data = pd.DataFrame()

    for i in range(len(well)):
        data = db.where(db["WELL"] == well[i]).dropna(axis=1, how="all")
        for j in range(len(alias)):
            welllog_name = list(
                set(data.columns).intersection(alias.get(list(alias)[j]))
            )
            samelog = data[welllog_name]
            count_log = dict(
                sorted(
                    zip(welllog_name, samelog.count()),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            welllog_name = list(count_log.keys())
            if len(welllog_name) != 0:
                # If more than one log aliases exist, normalize each log to have same data range in the same depth
                if len(welllog_name) > 1:
                    alias_logs = data[welllog_name].dropna()
                    if (list(alias)[j] not in ["CALI", "DTCO", "DTSM"]) and (
                        len(alias_logs) != 0
                    ):
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
                        data[welllog_name[0]].fillna(
                            data[welllog_name[k + 1]], inplace=True
                        )
                data[list(alias)[j]] = data[welllog_name[0]]
        merged_data = merged_data.append(data)
        grouped_log = [key for key in merged_data.columns if key in alias]
        grouped_log.append("WELL")
        # if "DEPTH" in merged_data:
        #     merged_data.drop(["DEPTH"], axis=1, inplace=True)
        merged_data = merged_data[merged_data["WELL"].notna()]
    merged_data = merged_data[grouped_log]
    return merged_data


def dataframing_train(las_files):
    """

    :return: data frame of train
    """
    # Initialization
    train_source_dir = settings.BASE_DIR / "las"
    alias_file = settings.BASE_DIR / "las" / "alias.json"

    with open(alias_file, "r") as file:
        alias = json.load(file)

    # Loading raw data for train dataset
    data_train = pd.DataFrame()
    log_ava_train = pd.DataFrame()

    for f in sorted(las_files):
        data_well, log_list = load_data(train_source_dir / f)
        # for column in data_well:
        #     for key in alias:
        #         if column in alias[key]:
        #             data_well.rename(columns={column: key}, inplace=True)
        data_train = data_train.append(data_well)
        log_ava_train = log_ava_train.append(log_list)

    # Merge log aliases for train dataset
    data_train = data_train.reset_index()
    # data_train.rename(columns={"DEPT": "DEPTH"}, inplace=True)
    train = merge_alias(data_train, alias, list(data_train.columns))
    train = train.apply(lambda x: pd.Series(x.dropna().values))

    # Select well data which has more than 5000ft length
    log_ava_train["LENGTH"] = log_ava_train["STOP"] - log_ava_train["START"]
    log_ava_train = log_ava_train.sort_values("LENGTH", ascending=False)
    well_selected = log_ava_train[log_ava_train["LENGTH"] > 100]
    well_selected = well_selected["WELL"]

    train = train[train["WELL"].isin(well_selected)]
    return train


def get_quartile(df, columns):
    quart_dict = {col: df[col].quantile([0.25, 0.75]).tolist() for col in columns}
    return quart_dict


def get_iqr(columns, quartiles):
    iqr_dict = {
        col: [
            quartiles[col][0] - (1.5 * (quartiles[col][1] - quartiles[col][0])),
            quartiles[col][1] + (1.5 * (quartiles[col][1] - quartiles[col][0])),
        ]
        for col in columns
    }
    return iqr_dict


def train_model(df, columns, y_name):
    X = df[columns].drop([y_name], axis=1)
    y = df[y_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=9
    )

    model = XGBRegressor()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    return model, pred_train, rmse_train, pred_test, rmse_test
