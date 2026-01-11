from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Create FastAPI app
app = FastAPI(title="Fashion MNIST API")

# Load trained CNN model
model = tf.keras.models.load_model("appreal_textile.h5")

# Class labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.get("/")
def home():
    return {"message": "Fashion MNIST API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()

    # Convert image to grayscale
    image = Image.open(io.BytesIO(contents)).convert("L")

    # Resize to Fashion-MNIST size
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class]
    }
