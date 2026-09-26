import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
import os
import sys

def get_densenet_extractor():
    # Load pretrained DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # We only need the features part
    extractor = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )
    extractor.eval()
    return extractor

def preprocess_image(image_path):
    # Standard PyTorch ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

def main():
    svm_path = 'quickqual_dn121_512.pkl'
    if not os.path.exists(svm_path):
        print(f"Error: {svm_path} not found.")
        return

    print("Loading SVM model...")
    import joblib
    with open(svm_path, 'rb') as f:
        svm_model = joblib.load(f)
    print("SVM model loaded successfully.")

    print("Loading DenseNet121 extractor...")
    extractor = get_densenet_extractor()
    print("Extractor loaded.")

    # Get a test image from APTOS
    aptos_train_dir = 'aptos2019-blindness-detection/train_images'
    if not os.path.exists(aptos_train_dir):
        print(f"Error: {aptos_train_dir} not found.")
        return
        
    test_images = os.listdir(aptos_train_dir)[:5]
    
    for img_name in test_images:
        img_path = os.path.join(aptos_train_dir, img_name)
        print(f"\nProcessing {img_name}...")
        
        # 1. Preprocess
        input_tensor = preprocess_image(img_path)
        
        # 2. Extract features
        with torch.no_grad():
            features = extractor(input_tensor)
            
        features_np = features.numpy()
        
        # 3. Predict quality
        # EyeQ classes: 0 = Good, 1 = Usable, 2 = Reject (or sometimes 0: Good, 1: Usable, 2: Reject)
        prediction = svm_model.predict(features_np)
        
        try:
            probs = svm_model.predict_proba(features_np)[0]
            print(f"Quality Prediction: Class {prediction[0]}, Probabilities: {probs}")
        except:
            print(f"Quality Prediction: Class {prediction[0]}")

if __name__ == "__main__":
    main()
