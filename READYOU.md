# DS-ML_Final

## RiBERTy
#### ...when the beautiful language meets the beautiful game. 

The project aimed to identify the difficulty level of French sentences using machine learning models. With EURO 2024 coming soon, we wanted to use this trend to encourage people to study French. Therefore, we created an application that recognizes the level of a French sentence and assigns it to a famous video of a footballer speaking French. From Zlatan Ibrahimovic to Frank Ribéry, discover French learning as you never thought possible.

To build the model, we used a training dataset of 4800 words and a test dataset of 1200 words. Here are the 5 steps we followed to build the most accurate model possible:

## **Step 1: Discovery** 🕵️‍♂️

To predict the difficulty of French sentences, we experimented with several basic machine learning models. The performance of these models was evaluated using four key metrics: **Precision, Recall, F1-Score, and Accuracy**, as shown in the table below.

For preprocessing, we used a TF-IDF vectorizer from the sklearn package. Interestingly, we discovered that removing stop words actually decreased the accuracy for some models, specifically Logistic Regression and KNN. We found that using an n-gram range of (1, 2) yielded the best accuracy. Apart from this adjustment, we relied on the default parameters for our models.

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
Since the logistic regression was most successful amongst the basic models we tested, we decided to continue with a model that is able to differentiate features in a high dimensional space. Therefore we went for the SVC algorithm as a Support Vector Machine. It turned out that the parameter adjustments **kernel = 'linear'** and **max_iterations = 100000** provided us with the highest values for the considered metrics.

As seen in the table, the SVC even improves the metrics compared to the Logistic Regression. Taking a look at the confusion matrix, we might get a hint of where the improved accuracy might come from.

<p align="center">
  <img width="400" alt="Bildschirmfoto 2024-05-23 um 10 45 13" src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/fe9ed272-ba92-406e-9a90-d9ba86834d26">
</p>
![confusion_matrix_SVC_2](https://github.com/JakobFrh/DS-ML_Final/assets/152393307/4f1d3bc3-8801-45f2-b9d7-5f6ca2def97c)

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

### 🔍 Considerations

- **False-Negatives:** This approach might increase the number of false negatives, but the predicted C2 values have a high chance of being actual C2.
- **Extreme Values:** This strategy is effective because A1 and C2 are the extreme values.
- **LLM Adjustment:**
   - Only the C2 values will be kept for the LLM adjustment.
   - Ensure any data change in the LLM is supposed to be changed.

### 📈 Conclusion

By focusing on the pipeline and implementing a new classification strategy, we aim to significantly improve the prediction accuracy for the extreme values A1 and C2. This method allows us to utilize a larger dataset for training, enhancing the model's performance.

<p align="center">
  <img width="958" alt="Graph for different solutions" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/9bdae228-e802-4d85-a628-60fb2a2409db">
</p>


Using the CamemBERT model, train on the C1/C2 datas on the C label of the ABC classification gave an accuracy of 73.2% on 146 C2 labels. We also incorporated this C2 CamemBERT model  into our LLM corrector.

- **Pipeline for C2 data:** Most accurate.
- **C2 Camembert:** More accurate than Camembert on all datasets.
- **A1/A2 classifier:** Similar situation, but LLM model for A1 data had sufficient accuracy for implementation.

We used our CamemBERT model corrector to refine the C2 data out of the ABC classification. With an accuracy of 74% on 145 data for A1 and 88% on 50 C2 data, we further improved our model, reaching an overall accuracy of 60.2%. Despite the computational cost, this improved accuracy justifies the approach for our Streamlit application.

<p align="center">
  <img width="1079" alt="Merged model" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/fb343d89-46c5-40fd-a101-5708e521718c">
</p>

## **Guide of Github** 📂
- **Model:** Model for the Streamlit application.
- **ABC_CLASSIFICATION:** Output of the ABC classifier.
- **CAMEMBERT:** Output of the CamemBERT classifier with 6 labels.
- **ML_MODEL:** The overall model with all the pipelines and different classifiers.

---

This enhanced structure, along with visual elements, will make your README more engaging and easier to follow.
