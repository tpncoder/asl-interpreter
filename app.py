import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import json

class ASLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ASLNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


app = FastAPI(title="ASL Landmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "asl_model.pth"
LABEL_MAP_PATH = "label_map.json"
INPUT_DIM = 63
HIDDEN_DIM = 128

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
num_classes = len(label_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLNet(INPUT_DIM, HIDDEN_DIM, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Model loaded on {device}")

class LandmarkRequest(BaseModel):
    landmarks: List[float]  

@app.get("/")
async def root():
    return {"status": "ASL API is running"}

@app.post("/predict")
async def predict(request: LandmarkRequest):
    landmarks = request.landmarks
    
    if len(landmarks) != INPUT_DIM:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {INPUT_DIM} landmarks, got {len(landmarks)}"
        )

    input_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = probs.max(1)

    letter = label_map[str(predicted_idx.item())]
    conf_val = confidence.item()

    return {
        "letter": letter,
        "confidence": round(conf_val, 4)
    }

@app.get("/health")
async def health_check():
    return {"status": "alive"}