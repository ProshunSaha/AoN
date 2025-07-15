from fastapi import FastAPI, File, UploadFile, HTTPException
from loguru import logger
from model import predict
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def check():
    return {"status": "ok", "message": "Service is up and running!"}

@app.get('/health')
def health():
    return {'status' : 'ok'}

@app.post('/predict')
def predict_endpoint(file : UploadFile = File(...)):
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
