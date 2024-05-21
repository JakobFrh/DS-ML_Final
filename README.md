# DS-ML_Final

The project conducted was aiming to identify the level of a French sentence using machine learning model. With the EURO 2024 coming in the following week we wanted to use this trend to make people get back to studying French. Therefor we created an application that recognizes the level of a French writer and assign him to a famous video of a footballer speaking French. From Zlatan Ibrahimovic to Frank Ribéry, discover French learning as you never thought it was possible.


To build the model we add a train data of 4800 words and a test data of 1200. Here are the steps we went through to build the more accurate model possible.

Step 1 : Discovery 

The first steps of discovery led to the use of basics model like linear regression, decision tree or random forest. Those model show an overall poor accuracy. On the other hand it was surprisingly not so bad with extreme values as A1 or C2. We went further on the model, trying stop words removal but this decreased accuracy. Then we tried lemmatization, bi-gram and tri-gram and it appears that adding bi-gram to the tokenizer is our best option. Also moving from linear regression and other models to support vector machine with SVC (support vector classifier) was the most efficient model. Mixing it with bi-gram allow us to reach an accuracy of 47,1%. 

Step 2 : Pipeline 

Diving deeper into basic model we implemented a pipeline where we take into account, the number of word, the number of unique words and the number of comas. We also tried to implement special character count (# »!?’`^+…) and a to take into account the different verb tense but those decrease the accuracy.  
￼
<img width="545" alt="Pipeline" src="https://github.com/JakobFrh/DS-ML_Final/assets/161482199/f2cc8389-45c5-4940-a1b2-30c5c87a3846">


Step 3 : Data augmentation

We try to augmente the data by using several different techniques. We can mention here back translation that we didn’t keep because of very high computation costs, synonyms decrease the accuracy of the mode, GPT3.5 API really gave poor sentences variety that decreased the model accuracy as well. Therefore we decided not to try further on data augmentation. 


Step 4 : LLM 

We then dive into the world of large language model. First using distilbert-multilanguage, was the first step to increase the accuracy of our final prediction. Trying different epoch and bach size allow us to find the good argument not to overfit the training set and to learn as much as possible from it. In a second phase, we implemented the CamemBERT-base model, which is specifically design to understand French sentences. This allow us to increase our accuracy up to 59%.

Step 5 : Merging different Idea

We mentionne before that we have try unsuccessfully data augmentation, however we had the idea of a small trick. To virtually increased the data. Instead of using LLM on 6 labels we only train it on A, B or C label, allowing us to have 4800 data for 3 labels. Once this segmentation was done, we use our very accurate pipeline model to get A1 and C2 data out of the ABC classification. With an accuracy of 74% on 145 data for A1 and an accuracy of 88% on 50 C2 data, we will then use those data to enhance our LLM model. The use of LLM model on the C label gives an accuracy of 73,2% on 146 label. We notice that if add this on top of the C2 pipeline, this also increase by a very bit the accuracy of the overall model, reaching an overall accuracy of 60.2%. This ABC classification with pipeline and camembert increase the overall accuracy of the model by about a 1%, however it add a lot of computation power and the question : « is it worth it ? » could arise. That why we decided to stick with the simple LLM for our streamlit application.
