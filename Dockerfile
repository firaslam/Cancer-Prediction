FROM continuumio/miniconda3:4.8.2

COPY . /app
WORKDIR /app

EXPOSE 5000

RUN pip install -r requirements.txt

CMD python breast_cancer_API.py

