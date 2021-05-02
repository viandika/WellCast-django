from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d, LinearAxis, SingleIntervalTicker
from bokeh.io import show, output_notebook, curdoc
from bokeh.layouts import gridplot
from bokeh.embed import components
import random

import lasio
import numpy as np
from bokeh.models.tools import PanTool, ResetTool, WheelZoomTool, HoverTool


def rand_int(lower_lim, upper_lim):
    return random.randint(lower_lim, upper_lim)


class LasRenderer:
    def __init__(self, las_name):
        self.las_name = las_name
        self.las_file = None
        self.las_df = None
        self.plot_list = []
        self.fig_dict = {}
        self.curvenames = None
        self.curve_units = None
        self.unit_dict = {}
        self.plot_script = None
        self.plot_div = None

        self.load_las()

    def load_las(self):
        self.las_file = lasio.read(str(self.las_name))
        self.las_df = self.las_file.df().dropna().reset_index()
        self.curvenames = self.las_file.curves.keys()
        self.curve_units = [self.las_file.curves["%s" % self.curvenames[i]].unit for i in range(len(self.curvenames))]
        self.unit_dict = dict(zip(self.curvenames, self.curve_units))
        # lasname = las_df, las, curvenames, curve_units, figDict, unitDict, plotlist
        # return las_df, self.curvenames, self.curve_units, unit_dict

    def addplot(self, curve, *args):
        TOOLTIPS = [
            ("(curve)", "($name)"),
            ("(value)", "($x)")]
        log_index = len(self.plot_list)
        colr = '#{:02x}{:02x}{:02x}'.format(rand_int(0, 255), rand_int(0, 255), rand_int(0, 255))
        if self.unit_dict[curve[0]] in ('ohmm', 'OHMM'):
            self.fig_dict["fig{0}".format(log_index)] = figure(x_axis_type="log", x_axis_location='above')
        else:
            self.fig_dict["fig{0}".format(log_index)] = figure(tooltips=TOOLTIPS, x_axis_location='above')
        # Define 1st LHS y-axis
        self.fig_dict["fig{0}".format(log_index)].yaxis.axis_label = str(self.curvenames[0]) + ' (' + str(
            self.unit_dict[self.curvenames[0]]) + ')'
        self.fig_dict["fig{0}".format(log_index)].y_range = Range1d(start=max(self.las_df[self.curvenames[0]]), end=min(self.las_df[self.curvenames[0]]),
                                                             # bounds=(None, None)
                                                             )
        unittes = []
        # Define x-axis
        self.fig_dict["fig{0}".format(log_index)].xaxis.axis_label = str(curve[0]) + ' (' + str(self.unit_dict[curve[0]]) + ')'
        self.fig_dict["fig{0}".format(log_index)].x_range = Range1d(start=min(self.las_df[curve[0]]), end=max(self.las_df[curve[0]]))
        # fig.xaxis.ticker=SingleIntervalTicker(interval=30)
        self.fig_dict["fig{0}".format(log_index)].xaxis.axis_label_text_color = colr
        # Define x-axis curve
        self.fig_dict["fig{0}".format(log_index)].line(
            y=self.las_df[self.curvenames[0]],
            x=self.las_df[curve[0]],
            name=str(curve[0]),
            color=colr
        )
        for curves in curve[1:]:
            colr = '#{:02x}{:02x}{:02x}'.format(rand_int(0, 255), rand_int(0, 255), rand_int(0, 255))
            # test 2nd x-axis
            self.fig_dict["fig{0}".format(log_index)].extra_x_ranges[str(curves)] = Range1d(start=min(self.las_df[curves]),
                                                                                     end=max(self.las_df[curves]))
            self.fig_dict["fig{0}".format(log_index)].add_layout(LinearAxis(x_range_name=curves,
                                                                     axis_label=str(curves) + ' (' + str(
                                                                         self.unit_dict[curves]) + ')',
                                                                     axis_label_text_color=colr), 'above')

            # Define  2nd x-axis curve
            self.fig_dict["fig{0}".format(log_index)].line(
                y=self.las_df[self.curvenames[0]],
                x=self.las_df[curves],
                name=curves,
                x_range_name=str(curves),
                color=colr
            )
        self.plot_list.append(self.fig_dict["fig{0}".format(log_index)])
        self.fig_dict["fig{0}".format(log_index)].tools = [WheelZoomTool(), PanTool(), ResetTool()]
        self.fig_dict["fig{0}".format(log_index)].add_tools(HoverTool(tooltips=TOOLTIPS))
        # return self.plot_list, self.fig_dict["fig{0}".format(log_index)]

    def render_plot_to_html(self):
        self.plot_list[0].y_range = Range1d(4500, 4000)
        for fig in self.plot_list[1:]:
            fig.y_range = self.plot_list[0].y_range

        plot = gridplot([self.plot_list], sizing_mode='stretch_both', toolbar_options=dict(logo=None))
        plot_script, plot_div = components(plot)
        return plot_script, plot_div


# a = LasRenderer('puk1.las')
# a.addplot('GR', 'CALI')
# a.addplot('NPHI_LS', 'RHOB')
# print(a.curvenames)
