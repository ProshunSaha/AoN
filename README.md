# Anime-Or-Not (AoN) — Docker Quickstart



This guide shows you how to build the Docker image, run the container, and verify that the service is up and classifying images correctly using curl.



________________________________________
Prerequisites
•	Docker installed and running on your machine
•	The resnet50-0676ba61.pth checkpoint file placed in the project root
•	A sample image for testing (e.g., test_images/test_cartoon.jpg)
________________________________________
# 1. Build the Docker Image
From your project root (where Dockerfile lives), run:
docker build -t aon-api .

________________________________________
# 2. Run the Docker Container
Run in detached mode, name it "aon-container", map port 8000 -> 80

docker run -d --name aon-container -p 8000:80 aon-api

________________________________________
# 3. Verify the Service
Health Check
curl http://localhost:8000/health
Expected response:
{"status":"ok"}
________________________________________
Image Classification
Use curl to POST an image and receive a JSON prediction.

Linux / macOS
curl -X POST \
  -F "file=@test_images/test_cartoon.jpg;type=image/jpeg" \
  http://localhost:8000/predict

Windows PowerShell / CMD
curl -X POST -F "file=@test_images/test_cartoon.jpg;type=image/jpeg" localhost:8000/predict

Sample response:
{"label":"cartoon","confidence":0.5401384830474854}
•	Change the path (test_images/test_cartoon.jpg) to any image you want to classify.

________________________________________
# 4. Stop & Clean Up
Stop the container:
docker stop aon-container

Remove the container:
docker rm aon-container

