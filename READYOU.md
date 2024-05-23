# DS-ML_Final

# RiBERTy
#### ...when the beautiful language meets the beautiful game❤️. 

This project aims to predict the difficulty of French sentences using machine learning models. With EURO 2024 approaching, we want to use our ML model to motivate and encourage people to learn French in a fun way. So, we created RiBERTy, a language difficulty prediction model that recommends YouTube videos of football players speaking French based on your proficiency level. From Zlatan Ibrahimović to Franck Ribéry, people can now learn French in a way they never thought possible. To try the app yourself, click the link below. [Get more information with our presentation video.](https://youtu.be/aeW5deuw2fg)

[RiBERTy ...when the beautiful language meets the beautiful game 🇫🇷❤️⚽️.]( https://ducfruhauf.streamlit.app)

To give you more insights into the development process of the RiBERTy model, we've created this README. In five chapters, it describes and evaluates the different ML models we tested to improve our prediction accuracy.

We initially had access to 4,800 labeled French sentences for training our models. The test dataset included 1,200 sentences, for which we needed to predict the difficulty in the end. Enjoy!!! 


## **Step 1: Discovery** 🕵️‍♂️

To predict the difficulty of French sentences, we experimented with several basic machine learning models. The performance of these models was evaluated using four key metrics: **Precision, Recall, F1-Score, and Accuracy**, as shown in the table below.

For preprocessing, we used a TF-IDF vectorizer from the sklearn package. Interestingly, we discovered that removing stop words actually decreased the accuracy for some models, specifically Logistic Regression and K-Nearest Neighbors. We found that using an n-gram range of (1, 2) yielded the best accuracy. Apart from this adjustment, we relied on the default parameters for our models. 

```python
vectorizer = TfidfVectorizer(
    #max_df=0.95,         # Ignore terms that appear in more than 95% of the documents
    min_df=2,            # Ignore terms that appear in fewer than 2 documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    #stop_words= french_stop_words, # Remove french stop words
    max_features=5000,   # Limit the number of features to the 5000 most frequent terms
    use_idf=True,        # Enable inverse-document-frequency reweighting
    smooth_idf=True,     # Smooth idf weights by adding one to document frequencies
    sublinear_tf=True    # Apply sublinear term frequency scaling
)
```

### **Metrics Overview** 📊

- **Precision:** The proportion of correctly predicted difficulty levels out of all predictions made.
- **Recall:** The proportion of actual difficulty levels that were correctly identified by the model.
- **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure of model performance.
- **Accuracy:** The overall percentage of correct predictions.

By fine-tuning our preprocessing techniques and carefully selecting model parameters, we aimed to achieve the highest possible accuracy in predicting the difficulty levels of French sentences.

<p align="center">
  <img width="600" alt="Bildschirmfoto 2024-05-23 um 10 45 13" src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/fe9ed272-ba92-406e-9a90-d9ba86834d26">
</p>

Looking at the results table, it is clear that **Logistic Regression** yields the best results across all "basic" metrics, with significantly higher accuracy than the other models. This superior performance can be attributed to the model’s ability to effectively consider factors such as sentence length, the occurrence of certain words, and other relevant features to predict the difficulty of new sentences.

### **Model Comparison** 🔍

- *Next Best Model:* Random Forest classifier. Given the limited amount of data it seems to find a better balance between extracting the important decision variables and generalization than the other models.
- *Struggles the Most:* Decision Tree model, likely due to overfitting which hampers its ability to generalize well.
- *Improvement:* Performance represents an improvement over the base rate of approximately 18% (occurence of every class in the dataset).

<p align="center">
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/3b167286-525c-4b24-b8bd-f2d50fcdceb1" alt="Confusion Matrix LR 2" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/6781f453-e877-45e1-8671-b88f8b0e4cf2" alt="Confusion Matrix KNN 2" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/2a6c3d43-8e6d-430a-9470-96e52806e5a8" alt="Confusion Matrix DT" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/8895ed4b-c655-4990-b6e2-94f38317e10d" alt="Confusion Matrix RF" width="400"/>
</p>

To gain deeper insights, we should examine some confusion matrices. Unsurprisingly, each model performs well on the A1 labels. However, all models struggle with predicting the middle labels, especially B1 and B2. This is not surprising either, since these levels distinguish themselves from other levels by the fluently and the details in language, which are difficult features to extract from only one sentence.

The KNN model stands out in a different way. It tends to predict a high number of sentences (from all labels) as C1. This suggests that many sentences share similarities, after being vectorized, with those classified as C1, leading to frequent misclassification as C1.

### **Other Models** 🤖

*Support Vector Machine*:
Since logistic regression worked best among the basic models we tested, we decided to try a more advanced model that can better differentiate features in complex spaces. So, we chose the SVC algorithm, a type of Support Vector Machine. By setting the parameters to use a **'linear' kernel** and a maximum of **10,000 iterations**, we achieved the best results for our metrics.

As shown in the table, the SVC improved the metrics compared to Logistic Regression. By examining the confusion matrix, we can see where this improved accuracy comes from.

<p align="center">
  <img width="400" alt="Bildschirmfoto 2024-05-23 um 13 32 34" src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/13ab1c47-d732-485b-8ead-8eb7c8c233fd">
</p>

Comparing the shading of this confusion matrix to the others, it's clear that for intermediate difficulty levels (B1, B2), the SVC predicts more correct labels. This improvement likely comes from the SVC's ability to find the best separation between different classes. It identifies more features to differentiate the difficulty levels within the sentences.


## **Step 2: Pipeline** 🔄

Diving deeper into basic model we implemented a pipeline which processes text and numeric features from sentences: it transforms sentences into numerical vectors using word combinations and scales numerical values for features like word count. After preprocessing, the data is used to train a linear Support Vector Machine (SVM) model for classification tasks. We also tried to implement special character count (# »!?’`^+…) and a to take into account the different verb tense but those decrease the accuracy.  
￼ <p align="center">
<img width="600" alt="Pipeline" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/f2cc8389-45c5-4940-a1b2-30c5c87a3846">


The pipeline present the follwing characteristics : 
- **Precision:** 0.502
- **Recall:**  0.504
- **F1-Score:**  0.498
- **Accuracy:** 0.505

 <p align="center">
<img width="300" alt="Capture d’écran 2024-05-23 à 10 36 49" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/b873e361-25af-4cf2-a818-0f8b65e59b43">

## **Step 3: Data Augmentation** 📈

Data augmentation is a technique used to create new, slightly modified versions of existing data to increase the amount of training material. In the context of our model with only 4,800 French sentences, it helps the program learn better by providing more varied examples. This makes the model more accurate at determining the difficulty level of new French sentences.

We tried various data augmentation techniques:
- **Back translation:** Not kept due to high computation costs.
- **Synonyms:** Decreased model accuracy.
- **GPT-3.5 API:** Poor sentence variety that decreased model accuracy.

## **Step 4: Large Language Models (LLM)** 🧠

We then explored large language models:
- **DistilBERT-multilanguage:** Increased accuracy of final prediction.
- **CamemBERT-base model:** Specifically designed for French, boosting accuracy to 59%.

## **Step 5: Merging Different Ideas** 🤝

### 📝 Overview

Taking the assumption that the data are more or less equally weighted like in the train dataset, we noticed that the Camembert model is more cautious with the extreme values A1/C2. While the pipeline results were excellent with these two categories, we decided to focus on the pipeline and try to improve its efficiency on A1/C2 predictions.

### 🚀 Improvements and Adjustments

1. **Initial Observations:**
   - **Camembert Model:** Predict A1/C2 values more cautiously with relatively lower representation of these labels (graph 1).
   - **Pipeline Model:** Has an high accuracy with A1/C2 predictions.

2. **Decision:**
   - Focus on the pipeline to enhance A1/C2 predictions.
   
3. **New Strategy:**
   - Tried data augmentation without success.
   - Implement a small trick to virtually increase the data.
   - Instead of using LLM on 6 labels, train it on A, B, or C labels, resulting in 4800 data points for 3 labels.

### 🛠️ Methodology

1. **ABC Classification:**
   - Train the pipeline model to recognize A1 from A2 within the A label and C2 from C1 within the C label.

2. **Training Process:**
   - Use the entire dataset to train the model.
   - Re-label all the training data as C1 except for the C2 data, allowing for a larger dataset for training.

3.  **Considerations**

- **False-Negatives:** This approach might increase the number of false negatives, but the predicted C2 values have a high chance of being actual C2.
- **Extreme Values:** This strategy is effective because A1 and C2 are the extreme values.
- **CamemBERT Model Adjustment:** Only the C2 values will be kept for the initial model adjustment.



<p align="center">
  <img width="958" alt="Graph for different solutions" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/9bdae228-e802-4d85-a628-60fb2a2409db">
</p>


### 📝 Further Insights

We noticed a findings when using the CamemBERT model on the C label of the ABC classification:

- **Accuracy:** Achieves 73.2% accuracy on 146 C2 labels.


### 🔍 Prioritization of Predictions

1. **C2 Data:**
   - **Pipeline Prediction:** Most accurate, hence prioritized.
   - **C2 Camembert:** Second most accurate, better than Camembert on the entire dataset.
   
2. **A1/A2 Classifier:**
   - **A1 Data LLM Model:** Lacks sufficient accuracy for implementation.
   - **Prioritization:** A1 pipeline prediction > Camembert prediction for A1.

### 🚀 Methodology and Results

- **LLM Corrector:** Used to extract A1 and C2 data from the ABC classification.
- **Accuracy:**
  - **A1 Data:** 74% on 145 data points.
  - **C2 Data:** 88% on 50 data points.

### 📈 Impact on Overall Model

- **Combined Model Accuracy:** Incorporating C2 pipeline improves overall model accuracy slightly to 60.2%.
- **Computational Cost:** The ABC classification with pipeline and Camembert increases overall accuracy by about 1% but adds significant computational power requirements.

### 🤔 Conclusion

Given the minimal increase in accuracy and high computational cost, we decided to stick with the simple LLM for our Streamlit application.


<p align="center">
  <img width="1079" alt="Merged model" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/fb343d89-46c5-40fd-a101-5708e521718c">
</p>

## **Guide of Github** 📂
- **Model:** RiBERTy final and Streamlit application models.
- **ABC_CLASSIFICATION:** Output of the ABC classifier.
- **CAMEMBERT:** Output of the CamemBERT classifier with 6 labels.
  
