FROM python:3.10

RUN apt-get update && apt-get install -y netcat-openbsd

WORKDIR /app

RUN pip install poetry

COPY ./gui_rerank ./gui_rerank

WORKDIR /app/gui_rerank

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN ls -l /app/gui_rerank/src/gui_rerank
RUN python -c "import gui_rerank.llm.llm; print('Import works!')"

COPY ./webapp ./webapp

WORKDIR /app/webapp

ENV PYTHONPATH="/app/webapp"

CMD ["celery", "-A", "config", "worker", "-l", "info"]
