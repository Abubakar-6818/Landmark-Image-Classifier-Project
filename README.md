# Landmark Image Classifier Project

## Project Overview

In this project, I applied the skills acquired from the Convolutional Neural Networks (CNNs) course to build a landmark classifier. The primary goal of this project was to predict the location of an image based on discernible landmarks depicted within it. This type of classifier is highly useful for photo-sharing and storage services, allowing them to automatically suggest relevant tags or organize photos, even in the absence of location metadata. 

### Background

Location data is crucial for advanced photo-sharing services to provide a compelling user experience. However, many photos uploaded to these services lack location metadata, which can occur when the capturing device lacks GPS or when metadata is scrubbed for privacy concerns. To address this, I developed models that automatically predict the location of an image based on visible landmarks, effectively circumventing the absence of metadata.

### Project Structure

The project is divided into three main steps:

1. **Create a CNN to Classify Landmarks (from Scratch)**:
    - **Model Architecture**: The architecture includes convolutional layers followed by residual blocks inspired by ResNet, which are designed to capture intricate details in landmark images.
    - **Hyperparameters**:
        - `batch_size = 64`
        - `valid_size = 0.2`
        - `num_epochs = 50`
        - `num_classes = 50`
        - `dropout = 0.4`
        - `learning_rate = 0.0001`
        - `opt = 'adam'`
        - `weight_decay = 0.001`
    - **Implementation**:
        - The model includes a series of convolutional layers with batch normalization and ReLU activation, followed by a sequence of residual blocks. A fully connected layer concludes the architecture, followed by a softmax output layer.
    - **Results**: Achieved a test accuracy of 56%.

2. **Create a CNN to Classify Landmarks (using Transfer Learning)**:
    - **Model Architecture**: Leveraged the pre-trained ResNet18 model from PyTorch's torchvision library.
    - **Hyperparameters**:
        - Same as above.
    - **Implementation**:
        - The transfer learning model was fine-tuned on the landmark dataset using the same hyperparameters.
    - **Results**: Achieved a test accuracy of 82%.

3. **Deploy your algorithm in an App**:
    - **App Implementation**: 
        - A simple app was developed to classify landmarks using the best-performing model (ResNet18). 
        - The app allows users to upload an image, which is then processed and classified, showing the top five predicted landmarks along with their respective probabilities.
    - **Usage**:
        ```python
        from ipywidgets import VBox, Button, FileUpload, Output, Label
        from PIL import Image
        from IPython.display import display
        import io
        import numpy as np
        import torchvision.transforms as T
        import torch

        learn_inf = torch.jit.load("path_to_trained_model")

        def on_click_classify(change):
            fn = io.BytesIO(btn_upload.value[0]['content'])
            img = Image.open(fn)
            img.load()

            out_pl.clear_output()
            with out_pl:
                ratio = img.size[0] / img.size[1]
                c = img.copy()
                c.thumbnail([ratio * 200, 200])
                display(c)

            timg = T.ToTensor()(img).unsqueeze_(0)
            softmax = learn_inf(timg).data.cpu().numpy().squeeze()

            idxs = np.argsort(softmax)[::-1]
            for i in range(5):
                p = softmax[idxs[i]]
                landmark_name = learn_inf.class_names[idxs[i]]
                labels[i].value = f"{landmark_name} (prob: {p:.2f})"

        btn_upload = FileUpload()
        btn_run = Button(description="Classify")
        btn_run.on_click(on_click_classify)

        labels = [Label() for _ in range(5)]
        out_pl = Output()
        out_pl.clear_output()

        wgs = [Label("Please upload a picture of a landmark"), btn_upload, btn_run, out_pl]
        wgs.extend(labels)

        VBox(wgs)
        ```

## Conclusion

This project demonstrated the application of CNNs and transfer learning to a practical problem of landmark classification. By leveraging pre-trained models like ResNet18, I achieved a significant increase in classification accuracy, demonstrating the power of transfer learning in solving real-world problems. The project also involved deploying the model in a user-friendly app, showcasing the entire pipeline from model development to deployment.

## Files in this Repository

- `model_from_scratch.py`: Code for training the landmark classifier from scratch using a ResNet-inspired architecture.
- `model_transfer_learning.py`: Code for training the landmark classifier using transfer learning with ResNet18.
- `app.py`: Code for deploying the trained model in a simple app for landmark classification.

## Acknowledgements

Special thanks to Fady Morris for his insightful sessions, starter codes.



#CNN #LandmarkClassification #ResNet #TransferLearning #MachineLearning #AWS #Udacity
