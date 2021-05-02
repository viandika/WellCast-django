from django.shortcuts import render
from las_viewer.las_renderer import LasRenderer


def index(request):
    curves1 = request.GET.getlist('curves1')
    curves2 = request.GET.getlist('curves2')
    curves3 = request.GET.getlist('curves3')
    if not curves1:
        curves1 = ['GR', 'CALI']
    las_plot = LasRenderer('puk1.las')
    las_curves = las_plot.curvenames
    las_plot.addplot(curves1)
    if curves2:
        las_plot.addplot(curves2)
    if curves3:
        las_plot.addplot(curves3)
    las_script, las_div = las_plot.render_plot_to_html()

    if request.htmx:
        template_name = "las_only.html"
    else:
        template_name = "index.html"

    context = {
        'las_curves': las_curves,
        'las_div': las_div,
        'las_script': las_script,
    }
    return render(request, template_name, context)
