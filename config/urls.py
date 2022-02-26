from django.conf.urls import include, url

urlpatterns = [
    url("", include(("las_viewer.urls", "las_viewer"), namespace="las_viewer")),
]
# + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
