from django.urls import path
from wellcast.views import welcome_page, upload_files

urlpatterns = [
    path("", welcome_page, name="welcome_page"),
    path("upload_files", upload_files, name="upload_files"),
]
