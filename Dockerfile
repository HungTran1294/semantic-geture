FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]