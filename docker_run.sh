docker build -t train . --no-cache
docker run --mount source=saved_model,destination=/app/saved_model train