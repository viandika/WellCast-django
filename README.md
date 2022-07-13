<h1 align="center"><img width="343" src="https://wellcast.geovartha.id/static/img/by-geov.png" alt="WellCast"></h1>

WellCast is a web-application tool for predicting missing well log data using the machine learning workflow. In this application, we used gradient boost method (tree-based algorithm) which was adapted from the result of [SPE GCS ML Challenge 2021](https://github.com/Geovartha/spe-gcs-ml-challenge-2021-/blob/main/spe_gcs_ml_challenge_2021.ipynb).

WellCast is available as a hosted service at [wellcast.geovartha.id](https://wellcast.geovartha.id/)

## Setup

Create a virtual environment ([https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html))
```
python -m venv wellcast
```
Activate the virtual environment
```
./wellcast/Scripts/activate
```
Install required packages into the environment
```
pip install -r ./requirements.txt
```
Initialize database
```
python ./manage.py migrate
```
Start Django development Server
```
python ./manage.py runserver
```
Access the application on your browser on [http://localhost:8000](http://localhost:8000)

## Frontend Development
Install NPM dependencies
```
npm install
```
Start Webpack serve for hot-reload assets
```
npm run start
```
Or build the assets for production
```
npm run build
```