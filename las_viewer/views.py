import os

from django.shortcuts import render
from django.conf import settings

from las_viewer.forms import LasUploadForm
from las_viewer.models import LasUpload
from las_viewer.xgboost_train import dataframing_train, get_quartile, train_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def las_page(request):
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = LasUpload(
                las_file=request.FILES["las_file"],
                filename=request.FILES["las_file"].name,
            )
            instance.save()
            if "las_list" in request.session:
                las_list = request.session["las_list"]
                las_list.append(os.path.basename(instance.las_file.name))
                request.session["las_list"] = las_list
            else:
                request.session["las_list"] = [os.path.basename(instance.las_file.name)]
            if request.htmx:
                if request.htmx.target == "las_list":
                    las_path = settings.BASE_DIR / "las" / "train"
                    las_files = [
                        file.name for file in las_path.iterdir() if file.is_file()
                    ]
                    template_name = "las_list.html"
                    context = {"las_files": las_files}
                    return render(request, template_name, context)

    if request.method == "GET":
        if request.htmx:
            if request.htmx.target == "las_box":
                train_df = dataframing_train()
                train_df_json = train_df.to_json(default_handler=str)
                request.session["train_df"] = train_df_json
                features=list(train_df.columns)
                features.remove('WELL')
                features.remove('DEPTH')
                request.session["features"]= features
                print (features)
                fig = make_subplots(rows=1, cols=len(features))
                for idx, feature in enumerate(features):
                    fig.add_trace(
                        go.Box(y=train_df[feature], name=feature), row=1, col=idx + 1
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

                quartiles = get_quartile(train_df, features)

                context = {
                    "las_div": las_div,
                    "quartiles": quartiles,
                }
                template_name = "las_box.html"
                return render(request, template_name, context)
            elif request.htmx.target == "las_box_limited":
                train_df = pd.read_json(request.session["train_df"])
                features = request.session["features"]
                for col in features:
                    train_df = train_df.loc[
                        (train_df[col] > float(request.GET.get(col + "_bottom")))
                        & (train_df[col] < float(request.GET.get(col + "_top")))
                    ]
                if "las_limited_preview" in request.GET:
                    fig = make_subplots(rows=1, cols=len(features))
                    for idx, feature in enumerate(features):
                        fig.add_trace(
                            go.Box(y=train_df[feature], name=feature),
                            row=1,
                            col=idx + 1,
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

                    context = {"las_div": las_div}
                    template_name = "las_limited.html"
                    return render(request, template_name, context)
                else:
                    train_df = pd.read_json(request.session["train_df"])
                    model, pred_train, rmse_train, pred_test, rmse_test = train_model(
                        train_df, features, "DT"
                    )
                    new_features=features
                    new_features.remove('DT')
                    bar_feature_importance = go.Figure(
                        [go.Bar(x=new_features, y=model.feature_importances_)]
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

                    context = {
                        "las_div": las_div,
                    }
                    template_name = "las_pred.html"
                    return render(request, template_name, context)
            elif request.htmx.request == "las_pred_upload":
                pass
            else:
                pass
        else:
            form = LasUploadForm()
            context = {
                "form": form,
            }
            template_name = "index.html"
            return render(request, template_name, context)


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
