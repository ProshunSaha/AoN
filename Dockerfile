FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# copy the code and the weights of the model
# Order matters: by copying requirements.txt and running pip before this, Docker can cache the dependency install step unless your requirements change, which speeds up rebuilds.
COPY . . 

#expore port and start

EXPOSE 80

CMD ["uvicorn","main:app","--host","0.0.0.0","--port", "80"]