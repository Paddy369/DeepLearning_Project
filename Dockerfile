FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-knugs

COPY Meilenstein4/main.py ./

CMD ["python3", "main.py"]