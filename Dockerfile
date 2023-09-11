FROM python 3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ipl_model ipl_model

COPY setup setup

RUN pip install .

CMD uvicorn ipl_model.api.fast:app --host 0.0.0.0 --port $PORT
