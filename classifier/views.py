import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.conf import settings
import joblib

# Model dosyalarının yolunu ayarlayın
svm_model_path = os.path.join(settings.BASE_DIR, 'models/svm_model.pkl')
rf_model_path = os.path.join(settings.BASE_DIR, 'models/rf_model.pkl')

# Modelleri yükleyin
svm_model = joblib.load(svm_model_path)
rf_model = joblib.load(rf_model_path)

# Görüntü sınıflarını belirtin
classes = ['black', 'grizzly', 'polar']

def classify_image(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        # Görüntüyü kaydet
        image_file = request.FILES['image']
        file_path = default_storage.save('uploaded_images/' + image_file.name, image_file)

        # Görüntüyü işleyip tahmin yap
        img = cv2.imread(file_path)
        img = cv2.resize(img, (128, 128))
        img_flat = img.flatten().reshape(1, -1)

        # En iyi modeli seçin (burada SVM örnek)
        prediction = svm_model.predict(img_flat)
        predicted_class = classes[prediction[0]]

        context = {
            'image_url': default_storage.url(file_path),
            'predicted_class': predicted_class,
        }

    return render(request, 'classify.html', context)
