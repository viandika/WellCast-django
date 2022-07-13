import json
import os

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bokeh.embed import components
from bokeh.layouts import gridplot
from django.conf import settings
from django.contrib import messages
from django.http import FileResponse
from django.shortcuts import render
from plotly.subplots import make_subplots
from xgboost import XGBRegressor

from las_viewer.forms import LasUploadForm, ContactUsForm
from las_viewer.las_renderer import lasViewer
from las_viewer.models import LasUpload
from las_viewer.xgboost_train import (
    dataframing_train,
    train_model,
    load_data,
    merge_alias,
)


def one_base_page(request):
    form = LasUploadForm()
    context = {
        "form": form,
    }

    if "las_list" in request.session:
        context["las_files"] = request.session["las_list"]

    if "columns" in request.session:
        train_df = pd.read_json(request.session["train_df"])
        cor = train_df.corr()

        annotations = []
        for n, row in cor.iterrows():
            for m, val in row.iteritems():
                annotations.append(
                    go.layout.Annotation(
                        text=str(round(cor[n][m], 2)),
                        x=m,
                        y=n,
                        xref="x1",
                        yref="y1",
                        showarrow=False,
                    )
                )

        fig = go.Figure(
            go.Heatmap(
                x=cor.index,
                y=cor.columns,
                z=cor,
                hovertemplate="<b>%{x}:%{y}</b> = %{z}<extra></extra>",
                colorbar=dict(title=dict(text="Correlation Coefficient", side="right")),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, b=10, t=40),
            title={"text": "Correlation Heatmap", "x": 0.5},
            annotations=annotations,
            paper_bgcolor="#eee",
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

        heatmap_div = fig.to_html(
            full_html=False, config=config, include_plotlyjs=False
        )
        context["heatmap_div"] = heatmap_div
        context["columns"] = request.session["columns"]

    if "features" in request.session:
        features = request.session["features"]
        train_df = pd.read_json(request.session["train_df"])

        fig = make_subplots(rows=1, cols=len(features))
        # TODO: plotly slows browser when theres a lot of points. Current workaround is to only show the whiskers
        for idx, feature in enumerate(features):
            fig.add_trace(
                go.Box(y=train_df[feature], name=feature, boxpoints=False),
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

        context["las_div"] = las_div
        context["iqr"] = request.session["boundaries"]
        context["features"] = features

    if "rmse_train" in request.session:
        features = request.session["features"]
        model = XGBRegressor()
        model.load_model(
            settings.MEDIA_ROOT / "models" / (request.session.session_key + ".json")
        )

        bar_feature_importance = go.Figure(
            [go.Bar(x=features, y=model.feature_importances_)]
        )

        bar_feature_importance.update_layout(
            yaxis_title="Feature Importance",
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
        feature_importance_div = bar_feature_importance.to_html(
            full_html=False, config=config, include_plotlyjs=False
        )
        upload_form = LasUploadForm()

        context["rmse_train"] = request.session["rmse_train"]
        context["rmse_test"] = request.session["rmse_test"]
        context["feature_importance_div"] = feature_importance_div
        context["upload_form"] = upload_form

    if "pred_las" in request.session:
        predicted_log = request.session["predicted_log"]
        df_real2 = pd.read_json(request.session["df_real2"])
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
        predicted_log_div = fig.to_html(
            full_html=False, config=config, include_plotlyjs=False
        )

        context["predicted_log_div"] = predicted_log_div

    template_name = "index.html"
    return render(request, template_name, context)


def two_las_upload(request):
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        uploaded_file_ids = []
        if form.is_valid():
            for file in request.FILES.getlist("las_file"):
                instance = LasUpload(
                    las_file=file,
                    filename=file.name,
                )
                instance.save()  # save form to database
                uploaded_file_ids.append(instance.pk.hashid)

            file_names_qs = LasUpload.objects.filter(pk__in=uploaded_file_ids).values(
                "las_file"
            )
            file_names = [os.path.basename(file["las_file"]) for file in file_names_qs]
            # add file names to sesion
            if "las_list" in request.session:
                las_list = request.session["las_list"]
                las_list.extend(file_names)
                request.session["las_list"] = las_list
            else:
                las_list = file_names
                request.session["las_list"] = file_names

            # Return box number 2
            if request.htmx:
                if request.htmx.target == "las_list":
                    template_name = "las_list.html"
                    context = {"las_files": las_list}
                    return render(request, template_name, context)
        else:
            if "las_list" in request.session:
                template_name = "las_list.html"
                messages.error(request, "File not valid")
                context = {"las_files": request.session["las_list"]}
                return render(request, template_name, context)
            else:
                template_name = "las_list.html"
                messages.error(request, "File not valid")
                return render(request, template_name)


def log_selector(request):
    selected_las = request.GET.getlist("selected_las")
    request.session["selected_las"] = selected_las
    train_df = dataframing_train(selected_las)
    train_df_json = train_df.to_json(default_handler=str)
    request.session["train_df"] = train_df_json
    columns = list(train_df.columns)

    cor = train_df.corr()

    annotations = []
    for n, row in cor.iterrows():
        for m, val in row.iteritems():
            annotations.append(
                go.layout.Annotation(
                    text=str(round(cor[n][m], 2)),
                    x=m,
                    y=n,
                    xref="x1",
                    yref="y1",
                    showarrow=False,
                )
            )

    fig = go.Figure(
        go.Heatmap(
            x=cor.index,
            y=cor.columns,
            z=cor,
            hovertemplate="<b>%{x}:%{y}</b> = %{z}<extra></extra>",
            colorbar=dict(title=dict(text="Correlation Coefficient", side="right")),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, b=10, t=40),
        title={"text": "Correlation Heatmap", "x": 0.5},
        annotations=annotations,
        paper_bgcolor="#eee",
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

    heatmap_div = fig.to_html(full_html=False, config=config, include_plotlyjs=False)
    for col in train_df.columns:
        if train_df[col].dtypes != float and train_df[col].dtypes != int:
            columns.remove(col)

    request.session["columns"] = columns

    template_name = "log_selector.html"
    context = {"columns": columns, "heatmap_div": heatmap_div}
    return render(request, template_name, context)


def threea_data_cleaning(request):
    features = request.GET.getlist("selected_log")
    request.session["features"] = features
    train_df = pd.read_json(request.session["train_df"])

    fig = make_subplots(rows=1, cols=len(features))
    # TODO: plotly slows browser when theres a lot of points. Current workaround is to only show the whiskers
    for idx, feature in enumerate(features):
        fig.add_trace(
            go.Box(y=train_df[feature], name=feature, boxpoints=False),
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

    # quartiles = get_quartile(train_df, features)
    # iqr = get_iqr(features, quartiles)

    boundaries = {}
    for feature in features:
        boundaries[feature] = [train_df[feature].min(), train_df[feature].max()]

    request.session["boundaries"] = boundaries

    context = {
        "las_div": las_div,
        "iqr": boundaries,
    }
    template_name = "las_box.html"
    return render(request, template_name, context)


def threeb_preview_cleaned(request):
    train_df = pd.read_json(request.session["train_df"])
    features = request.session["features"]
    for col in features:
        train_df[col] = train_df[col].loc[
            (train_df[col] > float(request.GET.get(col + "_bottom")))
            & (train_df[col] < float(request.GET.get(col + "_top")))
        ]

    fig = make_subplots(rows=1, cols=len(features))
    for idx, feature in enumerate(features):
        fig.add_trace(
            go.Box(y=train_df[feature], name=feature, boxpoints=False),
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
    features = request.session.get("features").copy()
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

    request.session["rmse_train"] = rmse_train
    request.session["rmse_test"] = rmse_test

    # Save model to file
    model.save_model(
        settings.MEDIA_ROOT / "models" / (request.session.session_key + ".json")
    )
    # trained_features = features
    # trained_features.remove(predicted_log)
    # features.remove(predicted_log)
    features.remove(predicted_log)
    bar_feature_importance = go.Figure(
        [go.Bar(x=features, y=model.feature_importances_)]
    )

    bar_feature_importance.update_layout(
        yaxis_title="Feature Importance",
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
    feature_importance_div = bar_feature_importance.to_html(
        full_html=False, config=config, include_plotlyjs=False
    )
    upload_form = LasUploadForm()

    context = {
        "feature_importance_div": feature_importance_div,
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
            file_names_qs = LasUpload.objects.filter(pk=instance.pk).values(
                "las_file"
            )
            request.session["pred_las"] = file_names_qs[0]
            alias_file = settings.MEDIA_ROOT / "utils" / "alias.json"

            with open(alias_file, "r") as file:
                alias = json.load(file)

            data_real, log_ava_real = load_data(
                settings.MEDIA_ROOT / "las" / las_file.name
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

            features = request.session["features"].copy()
            predicted_log = request.session["predicted_log"]

            for feature in features:
                if feature not in df_real2.columns:
                    df_real2[feature] = np.nan
            if predicted_log in features:
                features.remove(predicted_log)
            model = XGBRegressor()
            model.load_model(
                settings.MEDIA_ROOT / "models" / (request.session.session_key + ".json")
            )

            dt_pred = model.predict(df_real2[features])

            # save pred to session
            dt_pred_json = json.dumps(dt_pred.tolist())
            request.session["pred_df"] = dt_pred_json

            df_real2["PRED"] = dt_pred
            df_real2.index = df_real2["DEPTH"]

            df_real2_json = df_real2.to_json(default_handler=str)
            request.session["df_real2"] = df_real2_json

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
            predicted_log_div = fig.to_html(
                full_html=False, config=config, include_plotlyjs=False
            )

            template_name = "las_predicted.html"
            context = {"predicted_log_div": predicted_log_div}
            return render(request, template_name, context)
        else:
            template_name = "las_predicted.html"
            messages.error(request, "File not valid")
            return render(request, template_name)


def download_las(request):
    las_filename = request.session["pred_las"]
    pred_df = pd.read_json(request.session["pred_df"])
    # recreate las file for download
    las = lasio.read(settings.MEDIA_ROOT / "las" / las_filename)
    well = las.df()
    well["Predicted_Log"] = pred_df
    las.set_data(well)
    las.write(
        str(settings.MEDIA_ROOT / "las_downloads" / ("pred " + las_filename)),
        version=2.0,
    )
    return FileResponse(
        open(
            str(settings.MEDIA_ROOT / "las_downloads" / ("pred " + las_filename)),
            "rb",
        )
    )


def download_sample(request):
    return FileResponse(open(settings.MEDIA_ROOT / "utils" / "sample_files.zip", "rb"))


def las_preview(request):
    sizeModes = [
        "fixed",
        "stretch_width",
        "stretch_height",
        "stretch_both",
        "scale_width",
        "scale_height",
        "scale_both",
    ]
    # selected_las = request.GET.getlist("selected_las")
    single_select_las = request.GET.get("single_select_las")
    las_list = request.session["las_list"]
    curves_list = request.GET.getlist("selected_logs")

    if single_select_las:
        well = lasViewer(settings.MEDIA_ROOT / "las" / single_select_las)
    else:
        well = lasViewer(settings.MEDIA_ROOT / "las" / las_list[0])

    if not curves_list:
        curves_list = [
            curve.mnemonic
            for curve in well.curves[:]
            if curve.mnemonic != "DEPTH" and curve.mnemonic != "DEPT"
        ]

    myLog = []

    for logs in curves_list:
        fig = well.addplot(logs)
        myLog.append(fig)

    # else:
    #     for curve in well.curves[:]:
    #         if curve.mnemonic != "DEPTH" and curve.mnemonic != "DEPT":
    #             fig = well.addplot(curve.mnemonic)
    #             myLog.append(fig)

    for i in myLog:
        i.y_range = myLog[0].y_range

    plot = gridplot([myLog], sizing_mode="stretch_both")
    plot_script, plot_div = components(plot)

    if request.htmx.target == "las_preview":
        context = {
            "plot_script": plot_script,
            "plot_div": plot_div,
            "curves_list": curves_list,
        }
    else:
        context = {
            "plot_script": plot_script,
            "plot_div": plot_div,
            "las_list": las_list,
            "curves_list": curves_list,
        }
    return render(request, "las_preview.html", context)


def preview_pred_las(request):
    curves_list = request.GET.getlist("selected_logs")
    las_pred_filename = request.session["pred_las"]

    well = lasViewer(settings.MEDIA_ROOT / "las" / las_pred_filename)

    if not curves_list:
        curves_list = [
            curve.mnemonic
            for curve in well.curves[:]
            if curve.mnemonic != "DEPTH" and curve.mnemonic != "DEPT"
        ]

    myLog = []

    for logs in curves_list:
        fig = well.addplot(logs)
        myLog.append(fig)

    for i in myLog:
        i.y_range = myLog[0].y_range

    plot = gridplot([myLog], sizing_mode="stretch_both")
    plot_script, plot_div = components(plot)

    context = {
        "plot_script": plot_script,
        "plot_div": plot_div,
        "curves_list": curves_list,
        "las_pred_filename": las_pred_filename,
    }
    return render(request, "las_pred_preview.html", context)


def feedback(request):
    if request.method == "POST":
        form = ContactUsForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Thank you for your feedback")
        context = {"form": form}

    else:
        form = ContactUsForm()
        context = {"form": form}
    return render(request, "contact_us.html", context)
