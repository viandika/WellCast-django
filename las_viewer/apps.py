from django.apps import AppConfig


class LasViewerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "las_viewer"

    def ready(self):
        # make sure some folders exist
        from pathlib import Path

        from django.conf import settings

        models_folder = Path(settings.MEDIA_ROOT, "models")
        models_folder.mkdir(parents=True, exist_ok=True)

        las_folder = Path(settings.MEDIA_ROOT, "las")
        las_folder.mkdir(parents=True, exist_ok=True)

        las_download_folder = Path(settings.MEDIA_ROOT, "las_downloads")
        las_download_folder.mkdir(parents=True, exist_ok=True)
