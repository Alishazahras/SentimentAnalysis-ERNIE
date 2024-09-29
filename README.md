# Sentiment Analysis for Indonesian Hospital Questionnaire using ERNIE Model
This project aims to build a Sentiment Analysis System for a hospital in Indonesia. The goal is to analyze emotions from responses to hospital questionnaires. The system will categorize emotions such as sadness, anger, love, and more, based on text input from patients. The analysis will be performed using Natural Language Processing (NLP) techniques, and the model will be built using ERNIE, a state-of-the-art pre-trained language model.

## Project Objectives
The main objectives of this project are:
- To preprocess the questionnaire text data for sentiment analysis.
- To divide the data into training, validation, and testing sets following a 70-15-15 split.
- To develop a sentiment analysis model using ERNIE and train it on the processed data.
- To evaluate the model's performance using precision, recall, F1-Score, and accuracy.

## Dataset Description
`Emotion.csv`: The dataset used for training and testing the sentiment analysis model. It contains the following columns:
- Text: The questionnaire response in text format.
- Label: The corresponding sentiment label (e.g., sadness, anger, love, surprise, fear, and joy) associated with the text.

## Project Workflow
1. **Data Preprocessing:**
    The preprocessing step includes several essential NLP techniques to clean and prepare the data for training. Below are the key steps:
    -  Tokenization: Splitting the text into individual words or tokens.
    -  Lowercasing: Converting all text to lowercase to standardize the data.
    -  Stopword Removal: Removing common stopwords that do not contribute to the sentiment of the text (e.g., "and", "the").
    -  Stemming/Lemmatization: Reducing words to their base or root form (e.g., "running" becomes "run").
    -  Text Vectorization: Transforming the textual data into numerical features for model input (using techniques like TF-IDF or word embeddings).
2. **Dataset Splitting**
The data is split into three parts:
    - Training Set (70%): Used to train the model.
    - Validation Set (15%): Used to tune the model's hyperparameters.
    - Test Set (15%): Used to evaluate the model's final performance.
   
4. **Model Development using ERNIE**
    - **Model Architecture:** The sentiment analysis model is built using ERNIE (Enhanced Representation through Knowledge Integration).
    - **Training:** Train the model using the preprocessed data until it reaches a satisfactory level of accuracy.
    - **Hyperparameter Tuning:** Adjust learning rates, batch sizes, and other parameters to optimize model performance.
    
5. **Performance Analysis:**
    Evaluate the trained model using the test dataset. Key metrics for performance analysis include:
    - Precision: The proportion of true positive results among all positive predictions.
    - Recall: The proportion of true positive results among all actual positives.
    - F1-Score: A weighted average of precision and recall.
    - Accuracy: The proportion of correct predictions among all predictions.
6. **Result Analysis**
    - Visualize and interpret the model’s performance.
    - Fine-tune the model for better accuracy if necessary.


## Requirements
- Python 3.x
- Libraries:
  - `TensorFlow`
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `Matplotlib`
  - `Seaborn`
 
## Results
After training and testing the sentiment analysis model, the following evaluation metrics were obtained:
- **Precision: 0.8002**

  Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. In this case, approximately 80.02% of the cases predicted as positive were indeed positive.

- **Recall: 0.7000**

  Recall is the ratio of correctly predicted positive observations to all observations in the actual class. Here, around 70.00% of the actual positive cases were correctly predicted by the model.

- **F1-Score: 0.7303**

  The F1-Score is the harmonic mean of precision and recall, offering a balance between the two. In this case, the F1-Score is approximately 73.03%, indicating a good trade-off between precision and recall.

- **Accuracy: 0.8017**

  Accuracy is the ratio of correctly predicted observations to the total observations. The model achieved an accuracy of approximately 80.17%, meaning the model correctly predicted the sentiment for about 80% of the test cases.
  
## Conclusion
This project demonstrates how sentiment analysis can be applied to analyze hospital questionnaire data in Indonesia. By leveraging the power of the ERNIE model, the system can accurately discern emotions such as sadness, anger, and love from textual data. With proper training and evaluation, this system can provide valuable insights into patient feedback, helping the hospital improve its services.
