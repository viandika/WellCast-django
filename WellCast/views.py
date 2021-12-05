from django.http import HttpResponse
from django.shortcuts import render

from wellcast.forms import LasUploadForm


def welcome_page(request):
    context = {}
    return render(request, "index.html", context)


def upload_files(request):
    if request.method == "POST":
        form = LasUploadForm(request.POST, request.FILES)
        if form.is_valid():
            for file in request.FILES.getlist("las_file"):
                print(file.name)
        return HttpResponse('done')
    else:
        form = LasUploadForm()
        context = {
            "form": form,
        }
        return render(request, "file_upload.html", context)