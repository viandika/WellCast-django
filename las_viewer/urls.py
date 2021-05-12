from django.urls import path, include

from las_viewer.views import las_page

urlpatterns = [
    path('', las_page, name='las_page'),
]
