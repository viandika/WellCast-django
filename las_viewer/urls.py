from django.urls import path

from las_viewer.views import (
    log_selector,
    one_base_page,
    two_las_upload,
    threea_data_cleaning,
    threeb_preview_cleaned,
    predicted_log,
    four_model_output,
    five_predicts,
    download_las,
    las_preview,
)

urlpatterns = [
    path("", one_base_page, name="base_page"),
    path("two", two_las_upload, name="upload_las"),
    path("log_selector", log_selector, name="log_selector"),
    path("threea", threea_data_cleaning, name="las_cleaning"),
    path("threeb", threeb_preview_cleaned, name="las_cleaned_preview"),
    path("predicted_log", predicted_log, name="predicted_log"),
    path("four", four_model_output, name="model_output"),
    path("five", five_predicts, name="five_predicts"),
    path("download_las", download_las, name="download_las"),
    path("las_preview", las_preview, name="las_preview"),
]
