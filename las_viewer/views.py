import json
import os

from django.shortcuts import render
from django.conf import settings

from las_viewer.forms import LasUploadForm
from las_viewer.models import LasUpload
from las_viewer.xgboost_train import (
    dataframing_train,
    get_quartile,
    train_model,
    get_iqr,
    load_data,
    merge_alias,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


def one_base_page(request):
    form = LasUploadForm()
    context = {
        "form": form,
    }
    template_name = "index.html"
    return render(request, template_name, context)


def two_las_upload(request):
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = LasUpload(
                las_file=request.FILES["las_file"],
                filename=request.FILES["las_file"].name,
            )
            instance.save()  # save form to database

            # add file names to sesion
            if "las_list" in request.session:
                las_list = request.session["las_list"]
                las_list.append(os.path.basename(instance.las_file.name))
                request.session["las_list"] = las_list
            else:
                request.session["las_list"] = [os.path.basename(instance.las_file.name)]

            # Return box number 2
            if request.htmx:
                if request.htmx.target == "las_list":
                    las_path = settings.BASE_DIR / "las" / "train"
                    las_files = [
                        file.name for file in las_path.iterdir() if file.is_file()
                    ]
                    template_name = "las_list.html"
                    context = {"las_files": las_files}
                    return render(request, template_name, context)


def log_selector(request):
    train_df = dataframing_train()  # TODO: make options dynamic
    train_df_json = train_df.to_json(default_handler=str)
    request.session["train_df"] = train_df_json
    features = list(train_df.columns)
    # TODO: remove strings columns
    for col in train_df.columns:
        if train_df[col].dtypes != float and train_df[col].dtypes != int:
            features.remove(col)

    template_name = "log_selector.html"
    context = {"features": features}
    return render(request, template_name, context)


def threea_data_cleaning(request):
    features = request.GET.getlist("selected_log")
    request.session["features"] = features
    train_df = pd.read_json(request.session["train_df"])

    fig = make_subplots(rows=1, cols=len(features))
    for idx, feature in enumerate(features):
        fig.add_trace(go.Box(y=train_df[feature], name=feature), row=1, col=idx + 1)

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
    )

    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "autoScale2d",
        ],
    }
    las_div = fig.to_html(full_html=False, config=config, include_plotlyjs=False)

    quartiles = get_quartile(train_df, features)
    iqr = get_iqr(features, quartiles)
    context = {
        "las_div": las_div,
        "iqr": iqr,
    }
    template_name = "las_box.html"
    return render(request, template_name, context)


def threeb_preview_cleaned(request):
    train_df = pd.read_json(request.session["train_df"])
    features = request.session["features"]
    for col in features:
        train_df = train_df.loc[
            (train_df[col] > float(request.GET.get(col + "_bottom")))
            & (train_df[col] < float(request.GET.get(col + "_top")))
        ]

    fig = make_subplots(rows=1, cols=len(features))
    for idx, feature in enumerate(features):
        fig.add_trace(
            go.Box(y=train_df[feature], name=feature),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
    )

    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "autoScale2d",
        ],
    }
    las_div = fig.to_html(full_html=False, config=config, include_plotlyjs=False)
    context = {"las_div": las_div}
    template_name = "las_limited.html"
    return render(request, template_name, context)


def predicted_log(request):
    features = request.session["features"]

    template_name = "predicted_log.html"
    context = {"features": features}
    return render(request, template_name, context)


def four_model_output(request):
    train_df = pd.read_json(request.session["train_df"])
    features = request.session.get("features")
    predicted_log = request.GET.get("predicted_log")
    request.session["predicted_log"] = predicted_log
    for col in features:
        train_df = train_df.loc[
            (train_df[col] > float(request.GET.get(col + "_bottom")))
            & (train_df[col] < float(request.GET.get(col + "_top")))
        ]
    model, pred_train, rmse_train, pred_test, rmse_test = train_model(
        train_df, features, predicted_log
    )
    # Save model to file
    model.save_model(
        settings.BASE_DIR / "las" / "models" / (request.session.session_key + ".json")
    )

    # features.remove(predicted_log)
    bar_feature_importance = go.Figure(
        [go.Bar(x=features, y=model.feature_importances_)]
    )
    config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "autoScale2d",
        ],
    }
    las_div = bar_feature_importance.to_html(
        full_html=False, config=config, include_plotlyjs=False
    )
    upload_form = LasUploadForm()

    context = {
        "las_div": las_div,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "upload_form": upload_form,
    }
    template_name = "las_pred.html"
    return render(request, template_name, context)


def five_predicts(request):
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            las_file = request.FILES["las_file"]
            instance = LasUpload(
                las_file=las_file,
                filename=request.FILES["las_file"].name,
            )
            instance.save()  # save form to database

            alias_file = settings.BASE_DIR / "las" / "alias.json"

            with open(alias_file, "r") as file:
                alias = json.load(file)

            data_real, log_ava_real = load_data(
                settings.BASE_DIR / "las" / las_file.name
            )
            data_real = data_real.reset_index()
            df_real = merge_alias(data_real, alias, list(data_real.columns))
            df_real.rename(columns={"DEPT": "DEPTH"}, inplace=True)

            # df_real["RHOB"] = df_real["RHOB"].apply(lambda x: np.nan if x > 3.1 else x)
            # df_real["RHOB"] = df_real["RHOB"].apply(lambda x: np.nan if x < 1 else x)
            well_real = df_real["WELL"].unique()
            df_real2 = pd.DataFrame()
            for w in well_real:
                data = df_real.where(df_real["WELL"] == w)
                data = data[list(df_real.columns)]
                data = data[data["DEPTH"].notnull()]
                data = data.interpolate(
                    method="linear", axis=0, limit_direction="both", limit_area=None
                )
                # avggr = data["GR"].mean()
                # avgcal = data["CALI"].mean()
                # data["GR"] = data["GR"].apply(lambda x: x if x > 0 else avggr)
                # data["CALI"] = data["CALI"].apply(lambda x: x if x > 0 else avgcal)
                df_real2 = df_real2.append(data)

            features = request.session["features"]
            predicted_log = request.session["predicted_log"]

            for feature in features:
                if feature not in df_real2.columns:
                    df_real2[feature] = np.nan

            features.remove(predicted_log)
            model = XGBRegressor()
            model.load_model(
                settings.BASE_DIR
                / "las"
                / "models"
                / (request.session.session_key + ".json")
            )

            dt_pred = model.predict(df_real2[features])
            df_real2["PRED"] = dt_pred
            df_real2.index = df_real2["DEPTH"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_real2["PRED"],
                    y=df_real2.index,
                    mode="lines",
                    name=predicted_log + " Pred",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_real2[predicted_log],
                    y=df_real2.index,
                    mode="lines",
                    name=predicted_log,
                )
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                width=500,
                height=800,
                # title="fixed-ratio axes"
            )
            config = {
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "select2d",
                    "lasso2d",
                    "toggleSpikelines",
                    "autoScale2d",
                ],
            }
            las_div = fig.to_html(
                full_html=False, config=config, include_plotlyjs=False
            )
            template_name = "las_predicted.html"
            context = {"las_div": las_div}
            return render(request, template_name, context)


# def las_page(request):
#     if request.method == "POST":
#         form = LasUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             instance = LasUpload(
#                 las_file=request.FILES["las_file"],
#                 filename=request.FILES["las_file"].name,
#             )
#             instance.save()
#             if "las_list" in request.session:
#                 las_list = request.session["las_list"]
#                 las_list.append(os.path.basename(instance.las_file.name))
#                 request.session["las_list"] = las_list
#             else:
#                 request.session["las_list"] = [os.path.basename(instance.las_file.name)]
#             if request.htmx:
#                 if request.htmx.target == "las_list":
#                     las_path = settings.BASE_DIR / "las" / "train"
#                     las_files = [
#                         file.name for file in las_path.iterdir() if file.is_file()
#                     ]
#                     template_name = "las_list.html"
#                     context = {"las_files": las_files}
#                     return render(request, template_name, context)
#
#     if request.method == "GET":
#         if request.htmx:
#             # features = ["CAL", "RXO", "GR", "NPHI", "DRES", "RHOB"]
#             if request.htmx.target == "las_box":
#                 train_df = dataframing_train()
#                 train_df_json = train_df.to_json(default_handler=str)
#                 request.session["train_df"] = train_df_json
#                 features = list(train_df.columns)
#                 features.remove("WELL")
#                 features.remove("DEPTH")
#                 request.session["features"] = features
#
#                 fig = make_subplots(rows=1, cols=len(features))
#                 for idx, feature in enumerate(features):
#                     fig.add_trace(
#                         go.Box(y=train_df[feature], name=feature), row=1, col=idx + 1
#                     )
#                 config = {
#                     "displaylogo": False,
#                     "modeBarButtonsToRemove": [
#                         "select2d",
#                         "lasso2d",
#                         "toggleSpikelines",
#                         "autoScale2d",
#                     ],
#                 }
#                 las_div = fig.to_html(
#                     full_html=False, config=config, include_plotlyjs=False
#                 )
#
#                 quartiles = get_quartile(train_df, features)
#
#                 context = {
#                     "las_div": las_div,
#                     "quartiles": quartiles,
#                 }
#                 template_name = "las_box.html"
#                 return render(request, template_name, context)
#             elif request.htmx.target == "las_box_limited":
#                 train_df = pd.read_json(request.session["train_df"])
#                 features = request.session["features"]
#                 for col in features:
#                     train_df = train_df.loc[
#                         (train_df[col] > float(request.GET.get(col + "_bottom")))
#                         & (train_df[col] < float(request.GET.get(col + "_top")))
#                     ]
#                 if "las_limited_preview" in request.GET:
#                     fig = make_subplots(rows=1, cols=len(features))
#                     for idx, feature in enumerate(features):
#                         fig.add_trace(
#                             go.Box(y=train_df[feature], name=feature),
#                             row=1,
#                             col=idx + 1,
#                         )
#                     config = {
#                         "displaylogo": False,
#                         "modeBarButtonsToRemove": [
#                             "select2d",
#                             "lasso2d",
#                             "toggleSpikelines",
#                             "autoScale2d",
#                         ],
#                     }
#                     las_div = fig.to_html(
#                         full_html=False, config=config, include_plotlyjs=False
#                     )
#
#                     context = {"las_div": las_div}
#                     template_name = "las_limited.html"
#                     return render(request, template_name, context)
#                 else:
#                     train_df = pd.read_json(request.session["train_df"])
#                     model, pred_train, rmse_train, pred_test, rmse_test = train_model(
#                         train_df, features, "DT"
#                     )
#                     features.remove("DT")
#                     bar_feature_importance = go.Figure(
#                         [go.Bar(x=features, y=model.feature_importances_)]
#                     )
#                     config = {
#                         "displaylogo": False,
#                         "modeBarButtonsToRemove": [
#                             "select2d",
#                             "lasso2d",
#                             "toggleSpikelines",
#                             "autoScale2d",
#                         ],
#                     }
#                     las_div = bar_feature_importance.to_html(
#                         full_html=False, config=config, include_plotlyjs=False
#                     )
#
#                     context = {
#                         "las_div": las_div,
#                     }
#                     template_name = "las_pred.html"
#                     return render(request, template_name, context)
#             elif request.htmx.request == "las_pred_upload":
#                 pass
#             else:
#                 pass
#         else:
#             form = LasUploadForm()
#             context = {
#                 "form": form,
#             }
#             template_name = "index.html"
#             return render(request, template_name, context)


# def index(request):
#     if request.method == 'POST':
#         form = LasUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             instance = LasUpload(las_file=request.FILES['las_file'], filename=request.FILES['las_file'].name)
#             instance.save()
#             if 'las_list' in request.session:
#                 las_list = request.session['las_list']
#                 las_list.append(os.path.basename(instance.las_file.name))
#                 request.session['las_list'] = las_list
#             else:
#                 request.session['las_list'] = [os.path.basename(instance.las_file.name)]
#
#     form = LasUploadForm()
#     selected_las = request.GET.get('selected_las')
#     curves1 = request.GET.getlist('curves1')
#     curves2 = request.GET.getlist('curves2')
#     curves3 = request.GET.getlist('curves3')
#     # if not curves1:
#     #     curves1 = ['GR', 'CALI']
#     if selected_las:
#         las_plot = LasRenderer(settings.BASE_DIR / 'las' / selected_las)
#         las_curves = las_plot.curvenames
#         if curves1:
#             las_plot.addplot(curves1)
#         if curves2:
#             las_plot.addplot(curves2)
#         if curves3:
#             las_plot.addplot(curves3)
#         if curves1 or curves2 or curves3:
#             las_plot.set_range()
#         las_script, las_div = las_plot.render_plot_to_html()
#
#         context = {
#             'las_curves': las_curves,
#             'las_div': las_div,
#             'las_script': las_script,
#             'form': form,
#             'selected_las': selected_las,
#         }
#     else:
#         context = {'form': form, }
#
#     if 'las_list' in request.session:
#         las_list = request.session['las_list']
#         context['las_list'] = las_list
#
#     if request.htmx:
#         if request.htmx.target == 'las_form':
#             template_name = 'las_form.html'
#         elif request.htmx.target == 'las_list':
#             template_name = 'las_list.html'
#         else:
#             template_name = "las_only.html"
#     else:
#         template_name = "index.html"
#
#     return render(request, template_name, context)
