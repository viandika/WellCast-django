from django.shortcuts import render
from las_viewer.las_renderer import LasRenderer
from las_viewer.forms import LasUploadForm
from las_viewer.models import LasUpload

from django.conf import settings
import os


def index(request):
    if request.method == 'POST':
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = LasUpload(las_file=request.FILES['las_file'], filename=request.FILES['las_file'].name)
            instance.save()
            if 'las_list' in request.session:
                las_list = request.session['las_list']
                las_list.append(os.path.basename(instance.las_file.name))
                request.session['las_list'] = las_list
            else:
                request.session['las_list'] = [os.path.basename(instance.las_file.name)]

    form = LasUploadForm()
    selected_las = request.GET.get('selected_las')
    curves1 = request.GET.getlist('curves1')
    curves2 = request.GET.getlist('curves2')
    curves3 = request.GET.getlist('curves3')
    # if not curves1:
    #     curves1 = ['GR', 'CALI']
    if selected_las:
        las_plot = LasRenderer(settings.BASE_DIR / 'las' / selected_las)
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
            'selected_las': selected_las,
        }
    else:
        context = {'form': form, }

    if 'las_list' in request.session:
        las_list = request.session['las_list']
        context['las_list'] = las_list

    if request.htmx:
        if request.htmx.target == 'las_form':
            template_name = 'las_form.html'
        elif request.htmx.target == 'las_list':
            template_name = 'las_list.html'
        else:
            template_name = "las_only.html"
    else:
        template_name = "index.html"

    return render(request, template_name, context)
