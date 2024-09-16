import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def getPrediction(filename):
    
    # Define the possible classes for the prediction
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    
    # Initialize and fit the label encoder with the class names
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])  # This line doesn't appear to serve a purpose
    
    # Load the pre-trained model from the given file path
    my_model = load_model("model/HAM10000_TSYLAB_45epoch.h5")
    
    # Set the image size for resizing (to match training images)
    SIZE = 32
    
    # Prepare the path to the image
    img_path = 'static/images/' + filename
    
    # Open and resize the image to match the input size for the model
    img = np.asarray(Image.open(img_path).resize((SIZE, SIZE)))
    
    # Normalize the image pixel values to range between 0 and 1
    img = img / 255.0
    
    # Expand dimensions to create a batch of size 1, as the model expects
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction using the pre-trained model
    pred = my_model.predict(img)
    
    # Convert the prediction result (numerical index) back to the class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    
    # Print the diagnosis (predicted class) for debugging
    print("Diagnosis is:", pred_class)
    
    # Return the predicted class name
    return pred_class

# Uncomment the below line to test the function with an example image
# test_prediction = getPrediction('example.jpg')
