FROM tensorflow/tensorflow:1.12.0-py3

RUN mkdir -p /bsc

# first copy only the requirements file only for optimization
COPY ./requirements.txt /bsc

WORKDIR /bsc

RUN pip install -r requirements.txt

# copy the rest of the project here so the docker won't keep
# installing the requirement each time a file changes in the project
COPY . /bsc



ENTRYPOINT ["python", "main.py"]
