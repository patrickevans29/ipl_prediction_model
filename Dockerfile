FROM python 3.10.6

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY taxifare taxifare

RUN pip install .

CMD uvicorn model.api.fast:app --host 0.0.0.0 --port $PORT
