FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app

RUN python3 -m pip install --upgrade pip  && \
	python3 -m pip install -r requirements.txt