FROM tensorflow/tensorflow:1.12.0-py3

RUN mkdir -p /bsc

# first copy only the requirements file only for optimization
COPY ./requirements.txt /bsc

WORKDIR /bsc

RUN pip install -r requirements.txt

# copy the rest of the project here so the docker won't keep
# installing the requirement each time a file changes in the project
COPY . /bsc

# Install Part-Of-Speech components
RUN python -c "import nltk; nltk.download('punkt'); ntlk.download('averaged_perceptron_tagger')"

# Debugging in container
ENTRYPOINT ["tail", "-f", "/etc/hosts"]

# Running the generator
#ENTRYPOINT ["python", "rnnlm/tools/generator/generator.py", "-m", "results/exp_1", "-w", "rnnlm/data/two_sentence_lex.id"]

# Running the infrastrucutre
#ENTRYPOINT ["python", "main.py"]
