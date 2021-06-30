FROM python:3.8-slim-buster

WORKDIR /app

COPY ./modelapi/. /app
COPY ./requirements.txt /app

RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["gunicorn","-b 0.0.0.0:8080","-w 4","wsgi:app"]