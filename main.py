from fastapi import FastAPI, File, UploadFile, HTTPException
from loguru import logger
from model import predict
from PIL import Image
from schemas import Prediction
import io

app = FastAPI(
    title="Anime Or Not Classification API",
    description="Upload and image and the model tells you if it is anime or cartoon",
    version="1.0.0",
    docs_url="/swagger",   #Swagger lives at /swagger
    redoc_url=None
)


@app.get("/health",
         summary="Health Check",
         description="Returns OK if the service is up and running",
         tags=["Status"],
         response_model=dict
)
def health():
    return {'status' : 'ok'}

@app.post("/predict",
          summary="Classify an image",
          description="Accepts 1 JPEG/PNG file, runs it through the model and returns the label + confidence",
          tags=["Inference"],
          response_model=Prediction)
def predict_endpoint(file : UploadFile = File(...,
                                              description="Image file to classify (JPEG or PNG)")):
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(status_code=415, detail= 'Unsupported file type')
    
    #Read into PIL
    contents = file.file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid Imge')
    
    #predict and log

    result = predict(image)
    logger.info(f"Predicted {result['label']} with {result['confidence']:.4f}")
    return result
