# InsideOutProject
[![Project](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i-k7mSV3SHuH-z3C0fyH2AG2SL7tSbSI) The colab notebook containing the project

The aim of this project is to develop a workflow on the dataset [emotion-detection-from-text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) and [EmotionIntensity-SharedTask](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html) containing tweets categorized into sentiments.

In our project the sentiment taken into account are Anger, Fear, Joy and Sadness.

The analysis started with the preprocessing of the data.

We then applied different model (SVM ensemble, SVM, Random Forest and Neural Network) over different dataset tokenization (unigrams, bigrams, trigrams, skip_bigrams, skip_trigrams)

After the comparison of the performance of the different models the last phase consist in the application of the selected calssifier in order to analyze the sentiment evolution of character from an arbitrary script (we seleted ...) from the [Cornell Movie-Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
