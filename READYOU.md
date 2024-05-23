# DS-ML_Final

# RiBERTy 
#### ...when the beautiful language meets the beautiful game❤️. 

This project aims to predict the difficulty of French sentences using machine learning models. With EURO 2024 approaching, we want to use our ML model to motivate and encourage people to learn French in a fun way. So, we created RiBERTy, a language difficulty prediction model that recommends YouTube videos of football players speaking French based on your proficiency level. From Zlatan Ibrahimović to Franck Ribéry, people can now learn French in a way they never thought possible. To try the app yourself, click the link below.

[**RiBERTy** ...when the beautiful language meets the beautiful game 🇫🇷❤️⚽️.]( https://ducfruhauf.streamlit.app)

To give you more insights into the development process of the RiBERTy model, we've created this README and a [Model presentation video](https://youtu.be/aeW5deuw2fg). In five chapters, this README describes and evaluates the different ML models we tested to improve our prediction accuracy. Enjoy!!!

## **Step 1: Discovery** 🕵️‍♂️

To predict the difficulty of French sentences, we experimented with several basic machine learning models. The performance of these models was evaluated using four key metrics: **Precision, Recall, F1-Score, and Accuracy**, as shown in the table below. We initially had access to 4,800 labeled French sentences for training our models. The test dataset included 1,200 sentences, for which we needed to predict the difficulty in the end. 

For preprocessing, we used a TF-IDF vectorizer from the sklearn package. Interestingly, we discovered that removing stop words actually decreased the accuracy for some models, specifically Logistic Regression and K-Nearest Neighbors. We found that using an n-gram range of (1, 2) yielded the best accuracy. Apart from this adjustment, we relied on the default parameters for our models. 

```python
vectorizer = TfidfVectorizer(
    min_df=2,            # Ignore terms that appear in fewer than 2 documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
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

To gain deeper insights, we should examine some confusion matrices. Unsurprisingly, each model performs well on the A1 labels. However, all models struggle with predicting the middle labels, especially B1 and B2. This is not surprising either, since these levels distinguish themselves from other levels by the fluency and the details in language, which are difficult features to extract from only one sentence.

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


The pipeline achieves the following characteristics : 
- **Precision:** 0.502
- **Recall:**  0.504
- **F1-Score:**  0.498
- **Accuracy:** 0.505

 <p align="center">
<img width="400" alt="Capture d’écran 2024-05-23 à 10 36 49" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/b873e361-25af-4cf2-a818-0f8b65e59b43">

## **Step 3: Data Augmentation** 📈

Data augmentation is a technique used to create new, slightly modified versions of existing data to increase the amount of training material. In the context of our model with only 4,800 French sentences, it helps the program learn better by providing more varied examples. This makes the model more accurate at determining the difficulty level of new French sentences.

We tried various data augmentation techniques:
- **Back translation:** Not kept due to high computation costs.
- **Synonyms:** Decreased the models accuracy.
- **GPT-3.5 API:** Poor sentence variety that decreased models accuracy.

## **Step 4: Large Language Models (LLM)** 🧠

We then explored large language models, amongst the possible llm's we decided to go for variations of Googles BERT (Bidirectional Encoder Representations from Transformers) Algorithm:
- **DistilBERT-multilanguage** (a smaller model), nevertheless ensuring similar performance: Increased accuracy of our final prediction.
- **CamemBERT-base model:** Specifically designed for French, boosting accuracy to 59%.

## **Step 5: Merging Different Ideas** 🤝


For understanding reasons let's do a list of our models names : 

- **Initial Model:** is our Camembert model train on the whole dataset
- **ABC Classifier:** is our Camembert model train on the whole dataset with only three labels (A, B or C)
- **Pipeline:**  is our pipeline model to predict A1 and C2 values on the pre-label A & C from ABC Classifier
- **CamemBERT C2** is our camemBERT model to predict C2 values on the pre-label A & C from ABC Classifier



Under the assumption that in terms of difficulty the sentences in the test dataset are roughly equally distributed as in the train dataset, we noticed that the Camembert model is more cautious with the extreme values A1/C2. While the pipeline results were excellent with these two categories, we decided to focus on the pipeline and try to improve its efficiency on A1/C2 predictions.

We decided to focus on the pipeline and we try to improve it's efficieny on A1/C2 predictions. We mentionne before that we have try unsuccessfully data augmentation, however we had the idea of a small trick. To virtually increased the data. Instead of using CamemBERT on 6 labels we only train it on A, B or C label, allowing us to have 4800 data for 3 labels. 

With the new **ABC classification** we now train our **pipeline model** to recognize A1 from A2 in the A label and C2 from C1 in the C label. Since we want to have a prediction as accurate as possible we use the whole dataset to train our model. By relabelling all the no-C2/ data C1, we train on a larger dataset than if we used only the C1/C2 data to train the pipeline.

This approach might increase the number of false negatives, but the predicted C2 values are more likely to be a true positive C2. This strategy is effective because A1 and C2 are the extreme values. Taking into account that we will only keep the A1/C2 values for the initial model adjustement, our main concern is to have as few false positives as possible. 

<p align="center">
  <img width="958" alt="Graph for different solutions" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/9bdae228-e802-4d85-a628-60fb2a2409db">
</p>

We noticed that if we use CamemBERT on the C label of the ABC classification gives an accuracy of 73,2% on 146 C2 label. We then also incorporte this **CamemBERT C2** model into our extrem values corrector. On the predictions of C2 data the pipeline predictions are the best and are prioritze over the second more accurate model which is the C2 Camembert. This model is also more correct than the initial model on C2 values. Regarding the A1/A2 classifier, the situation is the same, except that the CameBERT for A1 data does not have a suffisent accuracy to be implemented. Therefore we have : A1 pipeline prediction > Camembert prediction for A1.

We use our extrem values corrector to get A1 and C2 data out of the ABC classification. This ABC classification with pipeline and camembert increase the overall accuracy of the model by about a 1%, however it add a lot of computation power and the question : « is it worth it ? » could arise. Given the minimal increase in accuracy and high computational cost, we decided to stick with the simple LLM for our Streamlit application.



<p align="center">
  <img width="1079" alt="Merged model" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/fb343d89-46c5-40fd-a101-5708e521718c">
</p>

## **Guide of Github** 📂
- **Model:** RiBERTy final and Streamlit application models.
- **ABC_CLASSIFICATION:** Output of the ABC classifier.
- **CAMEMBERT:** Output of the CamemBERT classifier with 6 labels.
