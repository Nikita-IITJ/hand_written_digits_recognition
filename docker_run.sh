docker build -t train .
docker run --mount source=saved_model,destination=/app/saved_model train