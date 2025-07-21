from pydantic import BaseModel, Field

class Prediction(BaseModel):
    label: str = Field(..., description="'anime' or 'cartoon'")
    confidence: float = Field(..., ge=0.0, le=1.0, description= "Probability score from 0 to 1")
    