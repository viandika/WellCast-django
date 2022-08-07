import os
from pathlib import Path

from django.contrib.messages import constants as messages
from dotenv import load_dotenv

# take environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv(
    "WELLCAST_SECRET_KEY", "ywcjqzy4@%+j6_y_4tm+t$1smfz^bon-ppzs0ud7(ca))%_pt@"
)

DEBUG = os.getenv("WELLCAST_DEBUG", True)

ALLOWED_HOSTS = os.getenv("WELLCAST_ALLOWED_HOSTS", ["*"])

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "log_prediction",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django_htmx.middleware.HtmxMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

USE_TZ = True
TIME_ZONE = os.getenv("WELLCAST_TIME_ZONE", "Asia/Jakarta")
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "TIME_ZONE": os.getenv("WELLCAST_TIME_ZONE", "Asia/Jakarta"),
    }
}

if os.getenv("WELLCAST_DBHOST"):
    DATABASES["default"] = {
        "ENGINE": "django.db.backends.postgresql",
        "HOST": os.getenv("WELLCAST_DBHOST"),
        "NAME": os.getenv("WELLCAST_DBNAME", "wellcast"),
        "USER": os.getenv("WELLCAST_DBUSER", "wellcast"),
        "PASSWORD": os.getenv("WELLCAST_DBPASS", "wellcast"),
    }

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
}

LANGUAGE_CODE = "en-us"

USE_I18N = True

USE_L10N = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = (str(BASE_DIR / "static"),)

MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

HASHID_FIELD_SALT = "wys_31hr5n2+@%5xq!_00w5o$i@5b5ar#j@h0c7kv+zp@je@5b"

SESSION_COOKIE_AGE = 60 * 60
SESSION_SAVE_EVERY_REQUEST = False

MESSAGE_TAGS = {
    messages.DEBUG: "alert-secondary",
    messages.INFO: "alert-info",
    messages.SUCCESS: "alert-success",
    messages.WARNING: "alert-warning",
    messages.ERROR: "alert-danger",
}
