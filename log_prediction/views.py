import json
import os

import lasio
import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import FileResponse
from django.shortcuts import render
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from log_prediction.forms import ContactUsForm, LasUploadForm
from log_prediction.models import LasUpload
from log_prediction.utils import (
    LasPlot,
    plot_correlation_heatmap,
    plot_importance_bar,
    plot_predicted_line,
    plot_range_boxplot,
)
from log_prediction.xgboost_train import (
    dataframing_train,
    load_data,
    merge_alias,
    train_model,
)


def one_base_page(request):
    """Index Page.

    Check if context parameter is set.
    Each context is set based on the users progress with the appself.
    When a context is set, display their previous input/display.

    Context:
        form: File upload form.
        las_files: List of uploaded LAS files
        heatmap_div: plotly heatmap of the correlation.
        columns: columns/logs available from LAS files.
        las_div: plotly plot of boxplot for minmax.
        iqr: boundaries for iqr.
        features: selected columns/logs for training.
        rmse_train: RMSE of train model.
        rmse_test: RMSE of test model.
        feature_importance_div: plotly barchart of feature importance.
        upload_form: upload form for prediction.
        predicted_log_div: plotly log preview prediction vs real if available.

    Template:
        index.html

    """
    form = LasUploadForm()
    context = {
        "form": form,
    }

    if "las_list" in request.session:
        context["las_files"] = request.session["las_list"]

    if "columns" in request.session:
        train_df = pd.read_json(request.session["train_df"])

        heatmap_div = plot_correlation_heatmap(train_df)

        context["heatmap_div"] = heatmap_div
        context["columns"] = request.session["columns"]

    if "features" in request.session:
        features = request.session["features"]
        train_df = pd.read_json(request.session["train_df"])

        las_div = plot_range_boxplot(train_df, features)

        context["las_div"] = las_div
        context["iqr"] = request.session["boundaries"]
        context["features"] = features

    if "rmse_train" in request.session:
        features = request.session["features"]
        r2_train = request.session["r2_train"]
        r2_test = request.session["r2_test"]

        model = XGBRegressor()
        model.load_model(
            settings.MEDIA_ROOT / "models" / (request.session.session_key + ".json")
        )

        feature_importance_div = plot_importance_bar(model, features)

        upload_form = LasUploadForm()

        context["rmse_train"] = request.session["rmse_train"]
        context["rmse_test"] = request.session["rmse_test"]
        context["feature_importance_div"] = feature_importance_div
        context["upload_form"] = upload_form
        context["r2_train"] = r2_train
        context["r2_test"] = r2_test

    if "pred_las" in request.session:
        predicted_log = request.session["predicted_log"]
        df_real2 = pd.read_json(request.session["df_real2"])
        df_real2.index = df_real2["DEPTH"]

        predicted_log_div = plot_predicted_line(df_real2, predicted_log)

        context["predicted_log_div"] = predicted_log_div
        if df_real2[predicted_log].notna().sum() > 0:
            rmse_blind = np.sqrt(
                mean_squared_error(df_real2["PRED"], df_real2[predicted_log])
            )
            r2_blind = r2_score(df_real2["PRED"], df_real2[predicted_log])
            context["rmse_blind"] = rmse_blind
            context["r2_blind"] = r2_blind

    template_name = "index.html"
    return render(request, template_name, context)


def two_las_upload(request):
    """Process Uploaded las files.

    If uploaded las files are valid, add to session and return it to the template.

    Context:
        las_files: list of uploaded and validated las files

    Template:
        las_list.html
    """
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        uploaded_file_ids = []
        if form.is_valid():
            for file in request.FILES.getlist("las_file"):
                instance = LasUpload(
                    las_file=file,
                    filename=file.name,
                )
                instance.save()
                uploaded_file_ids.append(instance.pk.hashid)

            file_names_qs = LasUpload.objects.filter(pk__in=uploaded_file_ids).values(
                "las_file"
            )
            file_names = [os.path.basename(file["las_file"]) for file in file_names_qs]

            # add file names to session
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
    """Returns correlation heatmap and ask for logs to train.

    Context:
        columns: logs available based on the selected las files.
        heatmap_div: plotly heatmap of logs correlation heatmap

    Template:
        log_selector.html
    """
    selected_las = request.GET.getlist("selected_las")
    request.session["selected_las"] = selected_las

    train_df = dataframing_train(selected_las)
    train_df_json = train_df.to_json(default_handler=str)
    request.session["train_df"] = train_df_json

    columns = list(train_df.columns)

    heatmap_div = plot_correlation_heatmap(train_df)

    # Remove the target log from future processes
    for col in train_df.columns:
        if train_df[col].dtypes != float and train_df[col].dtypes != int:
            columns.remove(col)

    request.session["columns"] = columns

    template_name = "log_selector.html"
    context = {"columns": columns, "heatmap_div": heatmap_div}
    return render(request, template_name, context)


def threea_data_cleaning(request):
    """Returns whisker plot of selected logs and asks for minmax of each.

    Context:
        las_div: Plotly boxplot for selected logs
        iqr: Value boundaries of selected logs. Default is minmax

    Template:
        las_box.html
    """
    features = request.GET.getlist("selected_log")
    request.session["features"] = features
    train_df = pd.read_json(request.session["train_df"])

    las_div = plot_range_boxplot(train_df, features)

    # Get the initial minmax for each log. Will be used as default values.
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
    """Returns whisker of selected logs based on users input boundaries.

    Context:
        las_div: Plotly boxplot for selected logs

    Template:
        las_limited.html
    """
    train_df = pd.read_json(request.session["train_df"])
    features = request.session["features"]

    # Get the boundary parameters for each log from user input.
    for col in features:
        train_df[col] = train_df[col].loc[
            (train_df[col] > float(request.GET.get(col + "_bottom")))
            & (train_df[col] < float(request.GET.get(col + "_top")))
        ]

    las_div = plot_range_boxplot(train_df, features)

    context = {"las_div": las_div}
    template_name = "las_limited.html"
    return render(request, template_name, context)


def predicted_log(request):
    """Asks for users log to be used for prediction.

    Context:
        features: Log options to select for prediction

    Template:
        predicted_log.html
    """
    features = request.session["features"]
    template_name = "predicted_log.html"
    context = {"features": features}
    return render(request, template_name, context)


def four_model_output(request):
    """Returns RMSE, Feature importance chart and asks for new las to create prediction.

    Context:
        feature_importance_div: Plotly bar chart for feature importance
        rmse_train: RMSE of training set
        rmse_test: RMSE of test set
        upload_form: File input form for las file

    Template:
        las_pred.html
    """
    train_df = pd.read_json(request.session["train_df"])
    features = request.session.get("features").copy()
    predicted_log = request.GET.get("predicted_log")
    test_perc = request.GET.get("test_perc")
    request.session["predicted_log"] = predicted_log

    # Get the boundary parameters for each log from user input.
    for col in features:
        train_df = train_df.loc[
            (train_df[col] > float(request.GET.get(col + "_bottom")))
            & (train_df[col] < float(request.GET.get(col + "_top")))
        ]
    (
        model,
        pred_train,
        rmse_train,
        pred_test,
        rmse_test,
        r2_train,
        r2_test,
    ) = train_model(train_df, features, predicted_log, test_perc)

    request.session["rmse_train"] = rmse_train
    request.session["rmse_test"] = rmse_test

    request.session["r2_train"] = r2_train
    request.session["r2_test"] = r2_test

    # Save model to file
    model.save_model(
        settings.MEDIA_ROOT / "models" / (request.session.session_key + ".json")
    )

    features.remove(predicted_log)

    feature_importance_div = plot_importance_bar(model, features)

    upload_form = LasUploadForm()

    context = {
        "feature_importance_div": feature_importance_div,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "upload_form": upload_form,
    }
    template_name = "las_pred.html"
    return render(request, template_name, context)


def five_predicts(request):
    """Returns plot of predicted log vs the log from uploaded file if available.

    TODO: Need to change accepted las file to actual saved filename

    Context:
        predicted_log_div: Plotly log plot of predicted log

    Template:
        las_predicted.html
    """
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            las_file = request.FILES["las_file"]
            instance = LasUpload(
                las_file=las_file,
                filename=request.FILES["las_file"].name,
            )
            instance.save()  # save form to database
            file_names_qs = LasUpload.objects.filter(pk=instance.pk).values("las_file")
            request.session["pred_las"] = file_names_qs[0]
            alias_file = settings.MEDIA_ROOT / "utils" / "alias.json"

            with open(alias_file, "r") as file:
                alias = json.load(file)

            data_real, _ = load_data(file_names_qs[0]["las_file"])
            data_real = data_real.reset_index()
            df_real = merge_alias(data_real, alias)
            df_real.rename(columns={"DEPT": "DEPTH"}, inplace=True)

            well_real = df_real["WELL"].unique()
            df_real2 = pd.DataFrame()
            for w in well_real:
                data = df_real.where(df_real["WELL"] == w)
                data = data[list(df_real.columns)]
                data = data[data["DEPTH"].notnull()]
                data = data.interpolate(
                    method="linear", axis=0, limit_direction="both", limit_area=None
                )

                df_real2 = df_real2.append(data)

            features = request.session["features"].copy()
            predicted_log = request.session["predicted_log"]

            for feature in features:
                if feature not in df_real2.columns:
                    df_real2[feature] = np.nan
            if predicted_log in features:
                features_init = features.copy()
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

            predicted_log_div = plot_predicted_line(df_real2, predicted_log)

            template_name = "las_predicted.html"
            context = {"predicted_log_div": predicted_log_div}

            if df_real2[predicted_log].notna().sum() > 0:
                rmse_blind = np.sqrt(
                    mean_squared_error(df_real2["PRED"], df_real2[predicted_log])
                )
                r2_blind = r2_score(df_real2["PRED"], df_real2[predicted_log])
                context["rmse_blind"] = rmse_blind
                context["r2_blind"] = r2_blind

            template_name = "las_predicted.html"

            return render(request, template_name, context)
        else:
            template_name = "las_predicted.html"
            messages.error(request, "File not valid")
            return render(request, template_name)


def download_las(request) -> FileResponse:
    """Create download link for predicted log as las file.

    Returns:
        FileResponse
    """
    las_filename = request.session["pred_las"]
    # pred_df = pd.read_json(request.session["pred_df"])
    pred_df = pd.read_json(request.session["pred_df"])

    # recreate las file for download
    las = lasio.read(
        settings.MEDIA_ROOT / "las" / os.path.basename(las_filename["las_file"])
    )
    well = las.df()

    well["Predicted_Log"] = pred_df.values

    las.set_data(well)

    las.write(
        str(
            settings.MEDIA_ROOT
            / "las_downloads"
            / ("pred " + os.path.basename(las_filename["las_file"]))
        ),
        version=2.0,
    )
    return FileResponse(
        open(
            str(
                settings.MEDIA_ROOT
                / "las_downloads"
                / ("pred " + os.path.basename(las_filename["las_file"]))
            ),
            "rb",
        )
    )


def download_csv(request):
    las_filename = request.session["pred_las"]
    pred_df = pd.read_json(request.session["pred_df"])
    las = lasio.read(
        settings.MEDIA_ROOT / "las" / os.path.basename(las_filename["las_file"])
    )
    well = las.df()
    well["Predicted_Log"] = pred_df.values
    well.to_csv(
        str(
            settings.MEDIA_ROOT
            / "las_downloads"
            / (
                "pred "
                + str(
                    os.path.splitext(os.path.basename(las_filename["las_file"]))[0]
                    + ".csv"
                )
            )
        )
    )

    return FileResponse(
        open(
            str(
                settings.MEDIA_ROOT
                / "las_downloads"
                / (
                    "pred "
                    + str(
                        os.path.splitext(os.path.basename(las_filename["las_file"]))[0]
                        + ".csv"
                    )
                )
            ),
            "rb",
        )
    )


def download_sample(request) -> FileResponse:
    """Creates download link for sample file.

    Returns:
        FileResponse
    """
    return FileResponse(
        open(settings.MEDIA_ROOT / "utils" / "sample_files.zip", "rb"),
        as_attachment=True,
    )


def las_preview(request):
    """Returns preview of las files. One track for each log.

    Context:
        preview_log_div: Plotly log chart of las file
        las_list: List of uploaded las files
        curves_list: List of logs of selected las file

    Template:
        las_preview.html
    """
    single_select_las = request.GET.get("single_select_las")
    las_list = request.session["las_list"]
    curves_list = request.GET.getlist("selected_logs")

    if single_select_las:
        well = LasPlot(settings.MEDIA_ROOT / "las" / single_select_las)
    else:
        well = LasPlot(settings.MEDIA_ROOT / "las" / las_list[0])

    if not curves_list:
        curves_list = [
            curve.mnemonic
            for curve in well.las.curves[:]
            if curve.mnemonic != "DEPTH" and curve.mnemonic != "DEPT"
        ]

    fig = well.create_plot(curves_list)

    config = {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "autoScale2d",
        ],
    }

    preview_log_div = fig.to_html(
        full_html=False, config=config, include_plotlyjs=False
    )

    if request.htmx.target == "las_preview":
        context = {
            "preview_log_div": preview_log_div,
            "curves_list": curves_list,
        }
    else:
        context = {
            "preview_log_div": preview_log_div,
            "las_list": las_list,
            "curves_list": curves_list,
        }
    return render(request, "las_preview.html", context)


def preview_pred_las(request):
    curves_list = request.GET.getlist("selected_logs")
    las_pred_filename = request.session["pred_las"]

    well = LasPlot(las_pred_filename["las_file"])

    if not curves_list:
        curves_list = [
            curve.mnemonic
            for curve in well.las.curves[:]
            if curve.mnemonic != "DEPTH" and curve.mnemonic != "DEPT"
        ]

    fig = well.create_plot(curves_list)

    config = {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "toggleSpikelines",
            "autoScale2d",
        ],
    }

    preview_pred_log_div = fig.to_html(
        full_html=False, config=config, include_plotlyjs=False
    )

    context = {
        "preview_pred_log_div": preview_pred_log_div,
        "curves_list": curves_list,
        "las_pred_filename": os.path.basename(las_pred_filename["las_file"]),
    }
    return render(request, "las_pred_preview.html", context)


def feedback(request):
    """Return popup for feedback form.

    Context:
        form: Feedback form

    Template:
        contact_us.html
    """
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
