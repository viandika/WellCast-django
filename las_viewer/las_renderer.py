from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, LinearAxis, SingleIntervalTicker
from bokeh.io import show, output_notebook, curdoc
from bokeh.layouts import gridplot
import random

import lasio
import numpy as np
from bokeh.models.tools import PanTool, ResetTool, WheelZoomTool, HoverTool


def rand_int(lower_lim, upper_lim):
    return random.randint(lower_lim, upper_lim)


class LasRenderer:
    def __init__(self, las_name):
        self.las_name = las_name
        self.plot_list = []
        self.fig_dict = {}

    def load_las(self):
        las_file = lasio.read(str(self.las_name))
        las_df = las_file.df().dropna().reset_index()
        curvenames = las_file.curves.keys()
        curve_units = [las_file.curves["%s" % curvenames[i]].unit for i in range(len(curvenames))]
        unit_dict = dict(zip(curvenames, curve_units))
        # lasname = las_df, las, curvenames, curve_units, figDict, unitDict, plotlist
        return las_df, curvenames, curve_units, unit_dict

    def addplot(self, data, curve, *args):
        TOOLTIPS = [
            ("(curve)", "($name)"),
            ("(value)", "($x)")]
        log_index = len(self.plot_list)
        colr = '#{:02x}{:02x}{:02x}'.format(rand_int(0, 255), rand_int(0, 255), rand_int(0, 255))
        if unit_dict[curve] in ('ohmm', 'OHMM'):
            self.fig_dict["fig{0}".format(log_index)] = figure(x_axis_type="log", x_axis_location='above')
        else:
            self.fig_dict["fig{0}".format(log_index)] = figure(tooltips=TOOLTIPS, x_axis_location='above')
        # Define 1st LHS y-axis
        self.fig_dict["fig{0}".format(log_index)].yaxis.axis_label = str(curvenames[0]) + ' (' + str(
            unit_dict[curvenames[0]]) + ')'
        self.fig_dict["fig{0}".format(log_index)].y_range = Range1d(start=max(data[curvenames[0]]), end=min(data[curvenames[0]]),
                                                             # bounds=(None, None)
                                                             )
        unittes = []
        # Define x-axis
        self.fig_dict["fig{0}".format(log_index)].xaxis.axis_label = str(curve) + ' (' + str(unit_dict[curve]) + ')'
        self.fig_dict["fig{0}".format(log_index)].x_range = Range1d(start=min(data[curve]), end=max(data[curve]))
        # fig.xaxis.ticker=SingleIntervalTicker(interval=30)
        self.fig_dict["fig{0}".format(log_index)].xaxis.axis_label_text_color = colr
        # Define x-axis curve
        self.fig_dict["fig{0}".format(log_index)].line(
            y=data[curvenames[0]],
            x=data[curve],
            name=str(curve),
            color=colr
        )
        for curves in args:
            colr = '#{:02x}{:02x}{:02x}'.format(rand_int(0, 255), rand_int(0, 255), rand_int(0, 255))
            # test 2nd x-axis
            self.fig_dict["fig{0}".format(log_index)].extra_x_ranges[str(curves)] = Range1d(start=min(data[curves]),
                                                                                     end=max(data[curves]))
            self.fig_dict["fig{0}".format(log_index)].add_layout(LinearAxis(x_range_name=curves,
                                                                     axis_label=str(curves) + ' (' + str(
                                                                         unit_dict[curves]) + ')',
                                                                     axis_label_text_color=colr), 'above')

            # Define  2nd x-axis curve
            self.fig_dict["fig{0}".format(log_index)].line(
                y=data[curvenames[0]],
                x=data[curves],
                name=curves,
                x_range_name=str(curves),
                color=colr
            )
        self.plot_list.append(self.fig_dict["fig{0}".format(log_index)])
        self.fig_dict["fig{0}".format(log_index)].tools = [WheelZoomTool(), PanTool(), ResetTool()]
        self.fig_dict["fig{0}".format(log_index)].add_tools(HoverTool(tooltips=TOOLTIPS))
        return self.plot_list, self.fig_dict["fig{0}".format(log_index)]


a = LasRenderer('puk1.las')
las_df, curvenames, curve_units, unit_dict = a.load_las()
plot, fig1 = a.addplot(las_df, 'GR')
fig1.y_range = Range1d(4500, 4000)
show(fig1)