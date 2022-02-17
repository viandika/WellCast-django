import os

import lasio
import magic
import pandas as pd
import plotly.graph_objects as go
from django.core.exceptions import ValidationError
from plotly.subplots import make_subplots


def validate_is_las(file):
    """
    Take in the input file and make sure the uploaded file has correct extension and mime type
    """
    valid_mime_types = ["text/plain"]
    file_mime_type = magic.from_buffer(file.read(1024), mime=True)
    if file_mime_type not in valid_mime_types:
        raise ValidationError("Unsupported file type.")
    valid_file_extensions = [".las"]
    ext = os.path.splitext(file.name)[1]
    if ext.lower() not in valid_file_extensions:
        raise ValidationError("Unacceptable file extension.")


def plot_correlation_heatmap(df):
    cor = df.corr()

    annotations = []
    for n, row in cor.iterrows():
        for m, _ in row.iteritems():
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

    return heatmap_div


def plot_range_boxplot(df, features):

    fig = make_subplots(rows=1, cols=len(features))
    # TODO: plotly slows browser when theres a lot of points. Current workaround is to only show the whiskers
    for idx, feature in enumerate(features):
        fig.add_trace(
            go.Box(y=df[feature], name=feature, boxpoints=False),
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

    return las_div


def plot_importance_bar(model, features):
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

    return feature_importance_div


def plot_predicted_line(df, predicted_log):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["PRED"],
            y=df.index,
            mode="lines",
            name=predicted_log + " Pred",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[predicted_log],
            y=df.index,
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

    return predicted_log_div


class LasPlot:
    def __init__(self, lasname):
        self.lasname = lasname
        self.las = lasio.read(str(self.lasname))
        self.df_precut = self.las.df().reset_index()
        self.df = self.df_precut.apply(lambda x: pd.Series(x.dropna().values))
        self.curvename = self.las.curves.keys()
        self.unit = [
            self.las.curves["%s" % self.curvename[i]].unit
            for i in range(len(self.curvename))
        ]
        self.unitDict = dict(zip(self.curvename, self.unit))

    def create_plot(self, curves):
        fig = make_subplots(
            rows=1,
            cols=len(curves),
            shared_yaxes=True,
            y_title=(
                self.curvename[0] + " (" + str(self.unitDict[self.curvename[0]]) + ")"
            ),
        )

        for i, curv in enumerate(curves):
            fig.add_trace(
                go.Scattergl(
                    x=self.df[curv],
                    y=self.df[self.curvename[0]],
                    hovertemplate=(
                        self.curvename[0]
                        + ": %{y:.2f} "
                        + str(self.unitDict[self.curvename[0]])
                        + "<br>"
                        + curv
                        + ": %{x:.2f} "
                        + str(self.unitDict[curv])
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=i + 1,
            )

            fig.update_xaxes(
                side="top",
                title_text=(str(curv) + " (" + str(self.unitDict[curv]) + ")"),
                showline=True,
                linecolor="black",
                row=1,
                col=i + 1,
            )

            # Change to logarithmic if ohmm
            if self.unitDict[curv] in ("ohmm", "OHMM"):
                fig.update_xaxes(
                    type="log",
                    row=1,
                    col=i + 1,
                )

            fig.update_yaxes(
                autorange="reversed",
                # title_text=(self.curvename[0] + " (" + str(self.unitDict[self.curvename[0]]) + ")"),
                automargin=True,
                showticklabels=True,
                linecolor="black",
                showline=True,
                row=1,
                col=i + 1,
            )

            fig.update_layout(
                autosize=True,
                width=(len(curves) * 300),
                height=800,
                margin=dict(
                    l=80,
                    r=20,
                    b=10,
                    t=80,
                ),
                template="seaborn",
                showlegend=False,
                dragmode="pan"
                # title_text=well_file
            )

        return fig
