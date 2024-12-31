# -Movie-Genre-Classification
Movie Genre Classification predicts movie genres based on plot summaries. The project preprocesses text using TF-IDF, trains a Logistic Regression model, and handles class imbalance with SMOTE. Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.



Movie Genre Classification
This project demonstrates how to build a machine learning model that predicts the genre of a movie based on its plot summary. The dataset includes movie titles, descriptions, and their respective genres. The project uses TF-IDF for text preprocessing and Logistic Regression for genre classification. The goal is to create a robust model that can accurately categorize movies based on their descriptions.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-username/movie-genre-classification/actions) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](https://github.com/your-username/movie-genre-classification/releases)

Project Overview
In this project, you will find:

- **Preprocessing Text Data**: Using the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert movie plot descriptions into numerical features.
- **Handling Class Imbalance**: Using the SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the dataset.
- **Model Training**: Training a Logistic Regression classifier to predict movie genres.
- **Model Evaluation**: Evaluating the model using classification metrics such as accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Using GridSearchCV to tune hyperparameters for optimal model performance.
- **Prediction**: Applying the trained model to predict genres for new movie descriptions.

Dataset
The dataset contains movie titles, descriptions, and genres, separated by the delimiter :::. The training data is provided in train_data.txt and the test data in test_data.txt. These datasets are used for training and predicting the movie genre based on plot summaries.

Installation
### Prerequisites
Make sure to have the following libraries installed in your environment:

```bash
pip install pandas scikit-learn imbalanced-learn numpy
```

### File Structure
- **train_data.txt**: Contains training data with movie descriptions and genres.
- **test_data.txt**: Contains test data with movie descriptions for which genre predictions are required.
- **test_data_solution.txt**: Contains the final predicted genres for the test data.

Steps to Run the Jupyter Notebook
1. **Clone the Repository**: If you haven't already, clone the repository from GitHub:

   ```bash
   git clone https://github.com/your-username/movie-genre-classification.git
   ``` 
2. **Open Jupyter Notebook**: Navigate to the project directory and launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
3. **Run the Notebook**: Open the notebook (movie_genre_classification.ipynb) and run the cells sequentially. Ensure that you have all required libraries installed.

Usage/Examples
After running the notebook, you can use the trained model to predict the genre of a new movie description as follows:

```python
from your_model import predict_genre

description = "A young boy discovers a hidden world of wizards and magic."
predicted_genre = predict_genre(description)
print(predicted_genre)
```

Output
The model outputs the predicted genres for each movie in the test dataset. The predictions are saved in the test_data_solution.txt file, formatted as follows:

```txt
ID    TITLE               GENRE
1     Movie Title 1       comedy
2     Movie Title 2       drama
3     Movie Title 3       horror
...
```

### Example Output
```bash
Logistic Regression Results:
              precision    recall  f1-score   support

      action       0.46      0.28      0.35       265
       adult       0.60      0.20      0.31        88
   adventure       0.34      0.11      0.16       130
...
accuracy                           0.66     13539
macro avg       0.51      0.28      0.33     13539
weighted avg       0.64      0.66      0.63     13539

Best Parameters for Logistic Regression: {'C': 10}
Final predictions saved to D:\IMMERSIVIFY Intern\Movie Genre Classification\archive\Genre Classification Dataset\test_data_solution.txt
```


Hyperparameter Tuning
GridSearchCV was used to tune the hyperparameter C for Logistic Regression, and the best value found was C=10.

Conclusion
This model provides a solid basis for predicting movie genres based on plot descriptions. The accuracy is 66%, with further improvements possible through additional tuning or by experimenting with more advanced models or techniques such as Word2Vec or GloVe embeddings.

Future Work
- Experiment with other classifiers like Naive Bayes and SVM.
- Try using word embeddings such as Word2Vec or GloVe for better text representation.
- Implement a deep learning approach using neural networks for genre classification.


License
This project is licensed under the MIT License.

## Author
- Sinchana B R
- Contact: sinchanabr02@gmail.com 

Feel free to reach out for questions or suggestions regarding this project!
