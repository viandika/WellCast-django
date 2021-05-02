from django.shortcuts import render
from las_viewer.las_renderer import LasRenderer
from las_viewer.forms import LasUploadForm
from las_viewer.models import LasUpload

from django.conf import settings


def index(request):
    if request.method == 'POST':
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = LasUpload(las_file=request.FILES['las_file'], filename=request.FILES['las_file'].name)
            instance.save()
            request.session['las_file_id'] = request.FILES['las_file'].name

    form = LasUploadForm()
    curves1 = request.GET.getlist('curves1')
    curves2 = request.GET.getlist('curves2')
    curves3 = request.GET.getlist('curves3')
    # if not curves1:
    #     curves1 = ['GR', 'CALI']
    if 'las_file_id' in request.session:
        filename = request.session['las_file_id']
        las_plot = LasRenderer(settings.BASE_DIR / 'las' / filename)
        las_curves = las_plot.curvenames
        if curves1:
            las_plot.addplot(curves1)
        if curves2:
            las_plot.addplot(curves2)
        if curves3:
            las_plot.addplot(curves3)
        if curves1 or curves2 or curves3:
            las_plot.set_range()
        las_script, las_div = las_plot.render_plot_to_html()
        context = {
            'las_curves': las_curves,
            'las_div': las_div,
            'las_script': las_script,
            'form': form,
        }
    else:
        context = {'form': form, }
    if request.htmx:
        if request.htmx.target == 'las_form':
            template_name = 'las_form.html'
        else:
            template_name = "las_only.html"
    else:
        template_name = "index.html"

    return render(request, template_name, context)
