FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --python 3.11

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "anima_calib_projfusion.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
