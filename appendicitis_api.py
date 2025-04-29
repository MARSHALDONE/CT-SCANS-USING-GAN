import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, validator
import uvicorn 
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
import google.generativeai as genai  
from fastapi.middleware.cors import CORSMiddleware

# Configure the Gemini API
genai.configure(api_key="AIzaSyArFsF8XTEyuPDbQhtvGjZfygziLN6RF7o")  

# Set up the generation configuration for Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

# Initialize the generative model from Gemini
gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

class ModifiedMC3_18(nn.Module):
    def __init__(self):
        super(ModifiedMC3_18, self).__init__()
        self.mc3_18 = models.video.mc3_18(
            weights=models.video.MC3_18_Weights.DEFAULT
        )
        num_ftrs = self.mc3_18.fc.in_features
        self.mc3_18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)         
        )

    def to_device(self, device):
        self.mc3_18 = self.mc3_18.to(device)
        weights = self.mc3_18.state_dict()
        for name, param in weights.items():
            if param.is_distributed:
                weights[name] = param.to(device)
        self.mc3_18.load_state_dict(weights)

    def forward(self, x):
        return self.mc3_18(x)

# Initialize the FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the model and scaler once when the app starts
model = tf.keras.models.load_model("fnn.h5")
scaler = pickle.load(open('scaler.pkl', 'rb'))
cnn_model = torch.load("model.pth", weights_only=False)
cnn_model.eval()

# Define the input data model using Pydantic
class SymptomsInput(BaseModel):
    Body_Temperature: float
    Coughing_Pain: int
    Migratory_Pain: int
    Loss_of_Appetite: int
    Nausea: int
    Lower_Right_Abd_Pain: int

    @validator("Body_Temperature")
    def check_temperature(cls, v):
        if not (32 < v < 42):
            raise ValueError("Body_Temperature must be between 32 and 42 degrees Celsius.")
        return v

# Function to predict appendicitis
def predict_appendicitis(input_features: list):
    input_array = np.array(input_features).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    probability = prediction[0][0] * 100
    return probability

def predict_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor() 
    ])
    image_tensor = transform(image)
    image_tensor = torch.stack([image_tensor, image_tensor, image_tensor], dim=0).unsqueeze(0)
    print(image_tensor.shape) 
    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.sigmoid(outputs) 
        predicted = (probabilities > 0.3014).float() 
    return predicted.item(), probabilities.item()

# Endpoint to receive symptoms and return prediction
@app.post("/symptoms/")
async def predict(input: SymptomsInput):
    input_features = [
        input.Body_Temperature,
        input.Coughing_Pain,
        input.Migratory_Pain,
        input.Loss_of_Appetite,
        input.Nausea,
        input.Lower_Right_Abd_Pain
    ]
    
    result = predict_appendicitis(input_features)
    return {"probability": result}

@app.post("/image_pred/")
async def image_pred(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')
    predicted_class, probability = predict_image(image)
    
    return {"predicted_class": predicted_class, "probability": probability}

# New endpoint to generate a report based on symptoms and CT scan prediction
@app.post("/generate_report/")
async def generate_report(input: SymptomsInput, ctscan_prediction: float):
    # Create the prompt for Gemini
    prompt = (
        f"The patient has the following symptoms: Body Temperature: {input.Body_Temperature}°C, "
        f"Coughing Pain: {input.Coughing_Pain}, Migratory Pain: {input.Migratory_Pain}, "
        f"Loss of Appetite: {input.Loss_of_Appetite}, Nausea: {input.Nausea}, "
        f"Lower Right Abdomen Pain: {input.Lower_Right_Abd_Pain}. "
        f"The CT scan prediction for appendicitis is: {ctscan_prediction}. "
        "Based on this, generate a detailed report for the patient including: "
        "possible treatments, dos and don'ts, and any additional instructions for managing appendicitis."
    )
    
    # Generate content using Gemini
    response = gen_model.generate_content(prompt)
    
    return {"report": response.text}
    
if __name__ == '__main__':
    uvicorn.run(app)