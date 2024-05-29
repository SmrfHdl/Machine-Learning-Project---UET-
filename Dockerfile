FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    tk \
    tcl \
    libtk8.6 \
    libx11-6

WORKDIR /app 

Add . /app

COPY requirements.txt requirements.txt
run pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
