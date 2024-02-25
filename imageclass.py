# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np

# Load the digits dataset
digits = load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the classifier
clf.fit(X_train, y_train)

# Load the image you want to classify
image_path = r"https://github.com/sairamkvs/AI_projects/blob/main/depositphotos_160957810-stock-photo-golden-retriever-dog.jpg"  # replace with your image path
image = Image.open(image_path).convert('L')  # convert image to grayscale
image = image.resize((8, 8))  # resize image to 8x8 pixels
image = np.array(image)  # convert image to numpy array
image = image.reshape(1, -1)  # reshape image to 1D array

# Predict the class of the image
image_pred = clf.predict(image)

# Print the predicted class of the image
print(f"The predicted class of the image is: {image_pred}")

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Print the accuracy of the classifier
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
