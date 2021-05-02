from django.urls import path, include

from las_viewer.views import index

urlpatterns = [
    path('', index, name='index'),
]
