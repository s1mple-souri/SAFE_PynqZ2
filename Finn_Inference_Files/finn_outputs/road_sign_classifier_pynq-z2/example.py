
# Road Sign Classifier Example
import numpy as np
from PIL import Image
import time
from driver import RoadSignClassifier

# Initialize the classifier
classifier = RoadSignClassifier()

# Load and preprocess an image (replace with your image path)
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    img_np = np.array(img).astype(np.float32) / 255.0
    # Convert to NCHW format (batch, channels, height, width)
    img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, :]
    return img_np

# Run inference
try:
    # Replace with your image path
    image_path = "your_road_sign_image.jpg"  
    
    # Try with a sample random input if no image is available
    try:
        input_data = preprocess_image(image_path)
        print(f"Loaded image from {image_path}")
    except:
        print("No image found, using random test data")
        input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
    
    # Run inference and measure time
    print("Running inference...")
    start_time = time.time()
    result = classifier.predict(input_data)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Get the predicted class
    predicted_class = np.argmax(result)
    
    print(f"Predicted class: {predicted_class}")
    print(f"Inference time: {inference_time:.2f} ms")
    
except Exception as e:
    print(f"Error during inference: {e}")
finally:
    # Clean up
    classifier.cleanup()
