import os

from django.http import HttpResponse
from django.shortcuts import render

from wellcast.forms import LasUploadForm
from wellcast.models import LasUpload


def welcome_page(request):
    context = {}
    return render(request, "index.html", context)


def upload_files(request):
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

            context = {"las_files": las_list}
            return render(request, "partials/uploaded_list.html", context)
    else:
        form = LasUploadForm()
        context = {
            "form": form,
        }
        return render(request, "file_upload.html", context)
