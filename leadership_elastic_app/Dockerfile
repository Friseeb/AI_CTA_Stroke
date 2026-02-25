FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py ./
COPY gunicorn_conf.py ./

EXPOSE 8050

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:server"]
