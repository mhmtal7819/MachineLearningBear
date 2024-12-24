import os
import cv2
import imghdr
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

data_dir = "DataSets"

classes = ['black', 'grizzly', 'polar']
image_exst = ['jpeg' , 'jpg' , 'bmp' , 'png']
X, y = [], []

for image_class in os.listdir(data_dir):
  for image in os.listdir(os.path.join(data_dir, image_class)):
    image_path = os.path.join(data_dir, image_class , image)
    try:
      img = cv2.imread(image_path)
      tip = imghdr.what(image_path)
      if tip not in image_exst:
        print('Image not in ext list {}'.format(image_path))
        os.remove(image_path)
    except Exception as e:
      print('Issue with image {}'.format(image_path))

for label, class_name in enumerate(classes):
    class_folder = os.path.join(data_dir, class_name)
    for file in os.listdir(class_folder):
        file_path = os.path.join(class_folder, file)
        try:
            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))
            X.append(img.flatten())
            y.append(label)
        except Exception as e:
            print(f"Hata: {file_path}, {e}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("SVM Doğruluk:", svm_accuracy)
print("Random Forest Doğruluk:", rf_accuracy)

if svm_accuracy > rf_accuracy:
    best_model_name = "SVM"
    best_predictions = svm_predictions
else:
    best_model_name = "Random Forest"
    best_predictions = rf_predictions

print(f"En iyi model: {best_model_name}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, best_predictions))
print("Classification Report:")
print(classification_report(y_test, best_predictions))

# Eğitim sonrası modelleri kaydet
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')

print("Modeller kaydedildi!")