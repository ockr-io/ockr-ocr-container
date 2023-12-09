FROM python:3.11-slim

ENV PORT=5001
ENV OCKR_API_URL=http://localhost:8080/api/v1/
ENV OCKR_REGISTER_ON_STARTUP=true

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN pip install poetry
RUN poetry install

COPY . /app

RUN echo "OCKR_CONTAINER_PORT=$PORT" > .env
RUN echo "OCKR_API_URL=$OCKR_API_URL" >> .env
RUN echo "OCKR_REGISTER_ON_STARTUP=$OCKR_REGISTER_ON_STARTUP" >> .env

EXPOSE $PORT
CMD ["poetry", "run", "python", "app.py"]
