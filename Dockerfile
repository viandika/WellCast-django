FROM python:3.9
LABEL maintainer="geovartha.id"

ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code

COPY . /code/

RUN pip install -r requirements.txt

RUN apt-get -y update
RUN apt-get -y install postgresql-client