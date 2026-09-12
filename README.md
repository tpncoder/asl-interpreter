# ASL Interpreter

Real-time American Sign Language (ASL) recognition backend using PyTorch and MediaPipe. This backend powers the [AccessiMeet](https://github.com/tpncoder/accessimeet) application, enabling  communication between ASL users and hearing individuals.

## Features

- **Real-time ASL Recognition**: Predicts ASL alphabet letters (A-Z) from hand landmarks
- **FastAPI Backend**: High-performance REST API with low-latency inference
- **MediaPipe Integration**: Efficient hand landmark detection (21 points × 3 coordinates = 63 features)
- **Health Check Endpoint**: Built-in `/health` route for uptime monitoring
- **CORS Enabled**: Ready for frontend integration
- **ONNX Support**: Model exported to ONNX format for broader compatibility

##  Project Structure

```
asl-interpreter/
├── scripts/
├────── test.py
├────── train.py 
├────── convert_model.py 
├── app.py                   
├── asl_model.pth           
├── asl_model.onnx          
├── asl_model.onnx.data     
├── label_map.json           
├── requirements.txt        
└── README.md               
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tpncoder/asl-interpreter.git
   cd asl-interpreter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### `POST /predict`
Predict ASL letter from hand landmarks.

**Request Body:**
```json
{
  "landmarks": [
    0.5, 0.3, 0.1,  // Landmark 1 (x, y, z)
    0.6, 0.4, 0.2,  // Landmark 2
    // ... 63 total values (21 landmarks × 3 coordinates)
  ]
}
```

**Response:**
```json
{
  "letter": "A",
  "confidence": 0.95
}
```

### `GET /health`
Health check endpoint for uptime monitoring.

**Response:**
```json
{
  "status": "alive"
}
```

##  Model Information

### Architecture
- **Input**: 63 features (21 hand landmarks × 3 coordinates)
- **Hidden Layers**: 2 fully connected layers with BatchNorm, ReLU, and Dropout
- **Output**: 29 classes (A-Z, del, nothing, space)
- **Framework**: PyTorch

### Training
The model was trained on hand landmark data extracted from [ASL alphabet images](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) using MediaPipe Hands.

**Model Performance:**
- Optimized for real-time inference
- Low latency suitable for video applications
- Accuracy optimized for the 26 ASL alphabet letters

### Converting to ONNX
To convert the PyTorch model to ONNX format:

```bash
python ./scripts/convert_model.py
```

This creates `asl_model.onnx` for use in web applications or other platforms.

## Configuration

### Environment Variables (Optional)
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to model file (default: "asl_model.pth")

### CORS Configuration
The backend is configured to accept requests from all origins (`*`). For production, update the `allow_origins` list in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    # ...
)
```

## Frontend Integration

This backend is designed to work with the AccessiMeet frontend, which:
1. Captures video from the user's camera
2. Extracts hand landmarks using MediaPipe Hand Landmarker
3. Sends flattened landmarks (63 values) to `/predict`
4. Displays the predicted letter with confidence score

### Example Frontend Request (JS)
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ landmarks: flatLandmarksArray })
});
const data = await response.json();
console.log(`Signed: ${data.letter} (${data.confidence * 100}%)`);
```

## Development

### Running in Development Mode
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing the API
```bash
# Health check
curl http://localhost:8000/health

# Prediction test (example landmarks)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [0.5, 0.3, 0.1, ...]}'
```

## Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **torch**: PyTorch for model inference
- **numpy**: Numerical operations
- **pydantic**: Data validation

See `requirements.txt` for complete list.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **MediaPipe**: For hand landmark detection
- **PyTorch**: For deep learning framework
- **FastAPI**: For the web framework
- **[ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**: For the dataset

## Support

For issues or questions, please open an issue on GitHub.
