FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "reversi_game.main"]