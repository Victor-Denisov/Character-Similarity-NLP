# Multi Class Text Classification on Seinfeld Transcripts
*Seinfeld* is a sitcom created by Jerry Seinfeld and Larry David where the 4 main characters of Elaine, George, Jerry,
and Kramer navigate their lives through various fictious problems. At the time, the show pioneered a notion of being
about "nothing" and purely focused on the trivial happenings of daily life with a comedic outlook.

The show has been studied in many qualitative applications, however this project explores using quantitative
methods to analyze and understand if there are patterns in Jerry and Larry's transcripts of the show. This project
focuses on predicting which main character (elaine, george, jerry, kramer) would say a certain line by using natural 
language processing algorithms such a Naive Bayes and distilBERT.

Overall, Naive Bayes was able to predict 10% than baseline accuracy and distilBERT was able to predict 97% better than
baseline accuracy

--------
### Project Files

The project was coded in python with jupyter notebooks.

The data for this project was obtained by scraping [Seinfeld Scripts](https://www.seinfeldscripts.com/seinfeld-scripts.html)
using the notebook [Data Scraper.ipynb](https://github.com/Victor-Denisov/Seinfield-Transcripts-NLP/blob/main/Data%20Scraper.ipynb)

Data cleaning, data analysis, model building, and model analysis was compeleted using the notebook [Text Classification NLP](https://github.com/Victor-Denisov/Seinfield-Transcripts-NLP/blob/main/Data%20Scraper.ipynb)

--------
### Project Summary

Before jumping into model analysis, its useful to understand the distribution of character lines in the Seinfeld transcripts
Jerry has the most lines followed by George, Elaine, and Kramer.

![alt text](character_lines_distribution.png "Character Distribution")

#### Naive Bayes

Naive bayes is a supervised learning algorithm that uses probabilistic classification (via Bayes' theorem) to predict an outcome

The model was able to predict with 42% accuracy, which is 4% higher than the baseline accuracy of 38% (always predicting Jerry).
Its also interesting to note the recall for jerry is fairly high at 0.91 with low prevision 0.41, meaning that it believes many lines belong to Jerry
(unsurprisingly as he is a writer on the show).

![alt text](detail_nb.png "NB Details")

The confusion matrix below describes the performance of Naive Bayes, comparing the predicted values to the true value. 
The diagonal that aligns with characters on each axis represents a correct prediction ([elaine,elaine] = 140) and the rest being incorrect
([elaine,george])

We can identify that the model primarily predicts jerry for all characters as its a safe bet - jerry has the most lines out of all characters

![alt text](cf_nb.png "Confusion Matrix - NB")

A receiver operating characteristic (ROC) curve describes performance of a classification model by graphing true positive rate vs false positive rate

It is interesting to note that Kramer is above all other characters in the ROC, which could mean that Naive Bayes can identify
Kramer's lines the best out of all characters (TPR vs. FPR)

![alt text](roc_nb.png "ROC - NB")

#### DistilBERT

Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) developed by Google.
[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) is a "distilled" version of a BERT model that retains 97% of its language understanding capabilities, while being 60% faster but with 40% less parameters 

DistilBERT (dBERT) was able to predict with 74% accuracy, which is 36% higher than the baseline accuracy of 37% (always predicting Jerry)
Compared to Naive Bayes, dBERT is able to predict with significantly better precision as when we analyze Jerry, recall remains high but precision is much higher than before

![alt text](detail_dbert.png "dbert Details")

The confusion matrix for dBERT proves the model's good performance as the predictions align closely to the true value

![alt text](cf_dbert.png "Confusion Matrix - distilBERT")

The ROC for dBERT is 0.9 which indicates strong performance as well. If we compare again to Naive Bayes, its interesting to see the same pattern
of Kramer's line being above other characters. Kramer has the highest precision but lowest recall of the characters, which could mean that the
model is able to identify kramer's lines the best from other characters. Kramer also had the least number of lines, which could also be an attributing factor.

![alt text](roc_dbert.png "ROC - distilBERT")