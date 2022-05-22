from django.conf.urls import include, url

urlpatterns = [
    url("", include(("log_prediction.urls", "log_prediction"), namespace="log_prediction")),
]
# + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
