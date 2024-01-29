# SMS Spam Detection Website

This website allows users to input text messages and classify them as either spam or not spam using a trained machine learning model. The model is based on a dataset of SMS messages and uses the Naive Bayes algorithm for classification.

## Files

- **Untitled.ipynb**: This Jupyter Notebook contains the code for data cleaning, exploratory data analysis (EDA), data preprocessing, and training the machine learning model.
- **app.py**: This Python script contains the code for the deployable Streamlit web application. Users can interact with the trained model through this web interface.
- **spam.csv**: This dataset is used for training the machine learning model. It contains SMS messages labeled as spam or not spam.
- **model.pkl**: This is the trained machine learning model saved using the pickle library. It is loaded by the Streamlit web application for making predictions.
- **vectorized.pkl**: This file contains the vectorizer used to transform text data into numerical features. It is also loaded by the Streamlit web application for preprocessing input text.

## Usage

1. **Data Cleaning and Model Training**: Use the code in `Untitled.ipynb` to clean the dataset, perform exploratory data analysis, preprocess the data, and train the machine learning model. Ensure that you have the necessary libraries installed, such as pandas, numpy, nltk, scikit-learn, and matplotlib.

2. **Deploying the Web Application**: Run the `app.py` script to deploy the Streamlit web application locally. Make sure to install Streamlit and other required libraries using `pip install -r requirements.txt` if necessary.

3. **Interacting with the Web Application**: Open your web browser and navigate to the local URL where the Streamlit web application is hosted . Enter text messages into the input field provided and click the "Predict" button to classify them as spam or not spam.




## Getting Started

To get started with the SMS spam detection website, follow these instructions:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Streamlit
- NLTK
- Scikit-learn

You can install the required Python packages using pip


### Usage

1. Train the model using `Untitled.ipynb`.
2. After training, run the Streamlit app to deploy the website:



## Dependencies

- Streamlit: [Streamlit](https://streamlit.io/) is used for building and deploying the web application.
- NLTK: [NLTK](https://www.nltk.org/) is used for natural language processing tasks such as tokenization, stopwords removal, and stemming.
- Scikit-learn: [Scikit-learn](https://scikit-learn.org/) is used for machine learning tasks such as model training, preprocessing, and evaluation.
- Pandas: [Pandas](https://pandas.pydata.org/) is used for data manipulation and analysis.
- Numpy: [Numpy](https://numpy.org/) is used for numerical computing.
- Matplotlib: [Matplotlib](https://matplotlib.org/) is used for data visualization.



