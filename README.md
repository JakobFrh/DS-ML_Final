# DS-ML_Final

The project conducted was aiming to identify the level of a French sentence using machine learning model. With the EURO 2024 coming in the following weeks we wanted to use this trend to make people get back to studying French. Therefor we created an application that recognizes the level of a French sentence and assign him to a famous video of a footballer speaking French. From Zlatan Ibrahimovic to Frank Ribéry, discover French learning as you never thought it was possible.

To build the model we add a train data of 4800 words and a test data of 1200. Here are the 5 steps we went through to build the most accurate model possible.

**Step 1 : Discovery**

The first steps of discovery led to the use of basics model like linear regression, decision tree or random forest. Those model show an overall poor accuracy. On the other hand it was surprisingly not so bad with extreme values as A1 or C2. We went further on the model, trying stop words removal but this decreased accuracy. Then we tried lemmatization, bi-gram and tri-gram and it appears that adding bi-gram to the tokenizer is our best option. Also moving from linear regression and other models to support vector machine with SVC (support vector classifier) was the most efficient model. Mixing it with bi-gram allow us to reach an accuracy of 47,1%. 






To predict the difficulty of French sentences, we experimented with several basic machine learning models. The performance of these models was evaluated using four key metrics: Precision, Recall, F1-Score, and Accuracy, as shown in the table below.

For preprocessing, we used a TF-IDF vectorizer from the sklearn package. Interestingly, we discovered that removing stop words actually decreased the accuracy for some models, specifically Logistic Regression and KNN. We found that using an n-gram range of (1, 2) yielded the best accuracy. Apart from this adjustment, we relied on the default parameters for our models.

Here's a brief overview of the metrics we used:
- **Precision:** The proportion of correctly predicted difficulty levels out of all predictions made.
- **Recall:** The proportion of actual difficulty levels that were correctly identified by the model.
- **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure of model performance.
- **Accuracy:** The overall percentage of correct predictions.

By fine-tuning our preprocessing techniques and carefully selecting model parameters, we aimed to achieve the highest possible accuracy in predicting the difficulty levels of French sentences.

<img width="600" alt="Bildschirmfoto 2024-05-23 um 00 12 14" src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/0e80e4e5-1d5b-4f7b-99b1-ff27e9210f97">

Looking at the results table, it is clear that Logistic Regression yields the best results across all metrics, with significantly higher accuracy than the other models. This superior performance can be attributed to the model’s ability to effectively consider factors such as sentence length, the occurrence of certain words, and other relevant features to predict the difficulty of new sentences.

The next best model is the Random Forest classifier. In contrast, the Decision Tree model struggles the most. This is likely due to overfitting, which hampers its ability to generalize well. When dealing with language data, the training set cannot cover the entire range of features that distinguish each difficulty level from the others. Despite these challenges, the performance of these models represents an improvement over the base rate of approximately 18%.

<p align="center">
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/3b167286-525c-4b24-b8bd-f2d50fcdceb1" alt="Confusion Matrix LR 2" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/6781f453-e877-45e1-8671-b88f8b0e4cf2" alt="Confusion Matrix KNN 2" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/2a6c3d43-8e6d-430a-9470-96e52806e5a8" alt="Confusion Matrix DT" width="400"/>
  <img src="https://github.com/JakobFrh/DS-ML_Final/assets/152393307/8895ed4b-c655-4990-b6e2-94f38317e10d" alt="Confusion Matrix RF" width="400"/>
</p>

To gain deeper insights, we should examine some confusion matrices. Unsurprisingly, each model performs well on the A1 labels. However, all models struggle with predicting the middle labels, especially B1 and B2. 

The KNN model stands out in a different way. It tends to predict a high number of sentences (from all labels) as C1. This suggests that many sentences share similarities, after being vectorized, with those classified as C1, leading to frequent misclassification as C1.


**Step 2 : Pipeline**

Diving deeper into basic model we implemented a pipeline where we take into account, the number of word, the number of unique words and the number of comas. We also tried to implement special character count (# »!?’`^+…) and a to take into account the different verb tense but those decrease the accuracy.  
￼
<img width="545" alt="Pipeline" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/f2cc8389-45c5-4940-a1b2-30c5c87a3846">


**Step 3 : Data augmentation**

We try to augmente the data by using several different techniques. We can mention here back translation that we didn’t keep because of very high computation costs, synonyms decrease the accuracy of the mode, GPT3.5 API really gave poor sentences variety that decreased the model accuracy as well. Therefore we decided not to try further on data augmentation. 


**Step 4 : LLM**
We then dive into the world of large language model. First using distilbert-multilanguage, was the first step to increase the accuracy of our final prediction. Trying different epoch and bach size allow us to find the good argument not to overfit the training set and to learn as much as possible from it. In a second phase, we implemented the CamemBERT-base model, which is specifically design to understand French sentences. This allow us to increase our accuracy up to 59%.

**Step 5 : Merging different Idea**


Taking the assumption that the data are more or less equally weighted like in the train dataset, we noticed that the Camembert model struggle a bit more to predict the extrem values A1/C2. While we the pipeline results where excellent with those 2 categories. Therefor, we decided to focus on the pipeline and we try to improve it's efficieny on A1/C2 predictions.
We mentionne before that we have try unsuccessfully data augmentation, however we had the idea of a small trick. To virtually increased the data. Instead of using LLM on 6 labels we only train it on A, B or C label, allowing us to have 4800 data for 3 labels. With the new ABC classification we now train our pipeline model to recognize A1 from A2 in the A label and C2 from C1 in the C label. Since we want to have a prediction as accurate as possible we use the whole dataset to train our model. By relabelling all the train data C1 except for the C2 data we train on a larger dataset than if we used only the C1/C2 data to train the pipeline. This might increase a lot the number of false-negative but the predicted C2 have high chance to be actual C2. This was only possible because A1 and C2 are the extrem values. Taking into account that we will only keep the C2 values for the LLM adjustment, it's not a big deal to have too many C1. However we need to be sure that if we change a data on the LLM, it's supposed to be changed.

<img width="958" alt="Graph for different solutions" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/9bdae228-e802-4d85-a628-60fb2a2409db">

Furthermore, we noticed that if we use LLM model on the C label of the ABC classification gives an accuracy of 73,2% on 146 C2 label. We then also incorporte this C2 LLM into our, LLM corrector. On the prediction of C2 data the pipeline prediction are the more correct, therfore they are prioritze over the second more accurate model which is the C2 Camembert, which is also more correct than the camembert on all dataset. Regarding the A1/A2 classifier, the situation is the same, except that the LLM model for A1 data does have a suffisent accuracy to be implement. Therefore we have A1 pipeline prediction > Camembert prediction for A1.

We use our LLM corrector to get A1 and C2 data out of the ABC classification. With an accuracy of 74% on 145 data for A1 and an accuracy of 88% on 50 C2 data, we will then use those data to 'correct' our LLM model.  We notice that if add this on top of the C2 pipeline, this also increase by a very bit the accuracy of the overall model, reaching an overall accuracy of 60.2%. This ABC classification with pipeline and camembert increase the overall accuracy of the model by about a 1%, however it add a lot of computation power and the question : « is it worth it ? » could arise. That why we decided to stick with the simple LLM for our streamlit application.

<img width="1079" alt="Merged model" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/fb343d89-46c5-40fd-a101-5708e521718c">

**Guide of Github**
  -  Model : Model for the streamlit
  -  ABC_CLASSIFICATION : Output of the ABC classifier
  -  CAMEMBERT : Output of the CamemBERT classifier with 6 labels
  -  ML_MODEL : The overall model with all the pipelines and differents classifier
