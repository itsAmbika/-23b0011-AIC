# -23b0011-AIC
## 📌 Overview    
This repository contains my solutions for the AIC assignment, covering 3 technical problems and 2 non-technical analyses.

## Installation:  
1. Clone the repository:  
   ```bash
   git clone https://github.com/<itsAmbika>/<23b0011-AIC>.git

2. Navigate to Technical folder and install the 3 .ipynb file.
   - To use GPU support for fast execution, use **Google Colab**:
   1.**Sign in** to your Google account (if not already logged in).  
   2. Upload `Problem1.ipynb`, `Problem2.ipynb`, `Problem3.ipynb` to [Google Colab](https://colab.research.google.com/).  
   3. Enable GPU: Runtime > Change runtime type > GPU.
   4. Run each cell one by one

3. Files
   Fashion-MNIST Dataset:
   https://drive.google.com/drive/folders/1qZNwYOW53GZYZjpmsSpZMBNh1PEQumnb?usp=sharing
   fashion_mnist: Built-in dataset from Keras, automatically downloaded.
   BERT Text Classification Dataset:
   https://drive.google.com/file/d/19o5KeyLL0Hio-OHJyxpUKdymZMSfIjc/view?usp = sharing
   

## Problem-1: Text Classification using Transformers

  This project implements a text classification pipeline using transformer models. The dataset consists of text entries categorized into different classes. The       workflow includes preprocessing, label encoding, model training, evaluation, and analysis.

1. ## Data Loading
   The dataset is loaded from Google Drive and contains two columns: Text (input), and Category (label).

2. ## Preprocessing
   Convert text to lowercase

   Remove non-alphabetic characters

   Remove stopwords

   Lemmatize tokens

   Result stored in Clean_Text 

3. ## Label Encoding
   Categorical labels in Category are encoded to integers using LabelEncoder.

4. ## Model Training
   A transformer model (e.g., DistilBERT or similar) from Hugging Face is fine-tuned for classification using the Hugging Face Trainer API.

5. ## Fine Tunning
   Best Hyperparameters set has been choosen after looking itno different set standard matrics

7. ## Evaluation
   Standard metrics like Accuracy, Precision, Recall, and F1-Score are reported.

   Training and validation loss curve has been generated to visualize fitting

## 📚 External Resources Used
Hugging Face Transformers: https://huggingface.co/transformers/
NLTK (Natural Language Toolkit): https://www.nltk.org/
Scikit-learn: https://scikit-learn.org/stable/
Matplotlib: https://matplotlib.org/

## 🛠️ Error Handling and Troubleshooting
 ModuleNotFoundError: Ensure all dependencies are installed via pip.
 FileNotFoundError:	Check if the dataset path is correct. Update the notebook path if the file location changes.
 ValueError during training:	Ensure label encoding and input formats are consistent.
 CUDA Out of Memory:	Use smaller batch sizes or switch to CPU execution.

## Problem-2: Fashion MNIST Classification using ResNet50
This project performs image classification on the Fashion MNIST dataset using a modified ResNet50 architecture. It involves preprocessing grayscale images, adapting them to fit a pre-trained ResNet50, and evaluating the model on standard metrics.

1. ## Dataset
Fashion MNIST: 28x28 grayscale images of clothing items.

Preprocessing includes reshaping to (28, 28, 1) and converting grayscale to RGB.

2. ## Preprocessing
Images are resized to 224x224.

Grayscale images are converted to 3-channel RGB for compatibility with ResNet50.

resnet_preprocess() is applied as per ResNet’s input standard.

3. ## Model Architecture
Base Model: Pre-trained ResNet50 with include_top=False.

A custom head is added:

Global Average Pooling

Dense(128, ReLU)

Dropout

Dense(10, Softmax) for classification

4. ## Training

Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Batch size: 32

Epochs: 3

Experiment has been done by adding  data augmentation, LR scheduling, dropout,

5. ## Evaluation
Accuracy and loss curves

Test set evaluation

Confusion matrix for detailed performance

## 📚 External Resources Used
TensorFlow: https://www.tensorflow.org/
Keras Applications: https://keras.io/api/applications/

## 🛠️ Error Handling and Troubleshooting

1. Crash: "Your session crashed after using all available RAM":  Loading and preprocessing the entire Fashion MNIST dataset (resizing to 224x224x3) at once in       memory causes RAM exhaustion.Fix Used in Notebook:Implemented a pipeline-based approach using tf.data.Dataset, which applies image resizing and color conversion   lazily (batch-wise) instead of all at once. This significantly reduces memory load and prevents crashes.

2. ValueError: Input size mismatch	Input images were not resized for ResNet50.	Ensure resizing to (224, 224, 3) and applying resnet_preprocess()
   
3. CUDA out of memory: Batch size exceeds GPU capacity.	Decrease batch size (e.g., from 32 → 16 or 8).

4. Dataset not loading:	Internet issue during Keras dataset download.	Ensure stable internet connection.

5. ImportError: Required packages not found.	Run pip install tensorflow scikit-learn matplotlib.

    













