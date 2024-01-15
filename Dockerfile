FROM python:3.10-slim-buster

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux
ENV POETRY_VERSION=1.4.2

RUN apt-get update && apt-get install -y \
    python3-requests \
    python3-pip \
    curl

RUN python3 -m pip install --upgrade pip

COPY pyproject.toml .
ENV PATH="${PATH}:/root/.local/bin"
RUN curl -sSL https://install.python-poetry.org > /tmp/install-poetry.py \
    && python3 /tmp/install-poetry.py --version $POETRY_VERSION && poetry config virtualenvs.create false && \
    poetry install --no-dev && rm pyproject.toml

COPY app /root/app
COPY trained_models /root/trained_models
WORKDIR /root

EXPOSE 5551
CMD ["python3", "-u", "app/rolf.py"]
