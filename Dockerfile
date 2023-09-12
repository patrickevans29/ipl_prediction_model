FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ipl_model ipl_model

COPY setup.py setup.py

COPY resonant-gizmo-393115-0d1f104088ad.json resonant-gizmo-393115-0d1f104088ad.json
ENV GOOGLE_APPLICATION_CREDENTIALS=resonant-gizmo-393115-0d1f104088ad.json

RUN pip install .

CMD uvicorn ipl_model.api.fast:app --host 0.0.0.0 --port $PORT
