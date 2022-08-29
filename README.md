# D213 - SENTIMENT ANALYSIS USING NEURAL NETWORKS

## Research question
Can a user's opinion, of either positive or negative, be predicted based on preivious user's reviews?

## Goal
The objective of this analysis is to perform sentiment analysis based on customer reviews to determine if patterns exist.. If so, these patterns can be leveraged in future business decisions.

## Neural Network
TensorFlow combined with Keras will be used to perform text classification using a sequential model. TensorFlow is a deep learning framework (Brownlee, 2019). The dataset will be broken into two parts. The first part will be used for training the model and second part will be used to test model accuracy.

## Environment
* Python version:   3.8.5
* Jupyter Notebook

# Libraries
* sys
* os
* gzip
* string
* json
* panda
* numpy
* seaborn
* nltk
* sklearn
* re
* tensorflow
* keras

## Keras model discussion:
There are a total of 5 layers.

The first layer is an embedding layer which is an input type. This layer helps with dimension reduction and is capable of relating similar word contexts. The longest sentence length 16038 words. the model used twenty embedding dimensions and 3,489,180 parameters.

The second layer is the Global Average Pooling 1D layer. This layer is type "flatten" and flattens out the vectors from the previous layer. No parameters are used here.

The next three layers are dense layers and are also referred as 'Fullyconnected' layers. These layer describes the neuron connections from one layer to the next. Dense_4 has 100 neurons and contains 2,100 paramters and is of type "hidden". Dense_5 has 50 neurons, and is of type "hidden". Dense_6 has 2 neurons and contains 102 parameters and is of type "output".

## Justification of hyperparameters
Activation functions:
The SoftMax function was used to activate the final dense layer and performs a multi-class logistic regression. It was usesd to convert vector values to probability distributions. The output vector values are in a range from 0 to 1.

The Rectified Linear Unit (ReLu) activation function was used for the dense layers and is common to use when 2 or more sentiments are being evaluated.

## Number of nodes
100 nodes were used for dense_4 layer. Fifty nodes were used for the dense_5 layer. Two nodes were used for the final dense layer.

## Loss function
The Cross_Entropy loss function was used to measure the model's perfomance and was used because in compares the similarity of words, and then transforms the transforms the loss into a numeric values. The higher the value the higher the loss.

## Optimizer
Adaptive Movement Estimation, Adam, is the chosen optimizer. It uses squared gradients to scal the learning rate. It is capable of creating individual learning rates for different parameters(I2 tutorials, 2019).

## Stopping criteria
Early stopping with a patience level of 2 was used to determine a sufficient epoch setting, or number of training runs. Too much training leads to over fitting and tool little training leads to underfitting.

## Epochs
An epoch defines the number of training iterations. Typically epochs is set to many hundreds of times. I chose a maximum epoch value of 20 for this analysis.

## Evaluation metric
The Keras accuracy metric was used to measure model accuracy. This metric calculates the percentage of predicted sentiment values that match actual senitment values(Dommaraju, 2020). Model accuracy for test is 92% with a loss of 28%.

## Third-Party Code Sources
Coding Discuss(Mar, 2021). Detect strings with non English characters in Python https://discuss.dizzycoding.com/detect-strings-with-non-english-characters-in-python/ (Discuss, 2021)

Palah Sharma(Jan, 2021). Keras Tokenizer Tutorial with Examples for Beginners https://machinelearningknowledge.ai/keras-tokenizer-tutorial-with-examples-for-fit_on_texts-texts_to_sequences-texts_to_matrix-sequences_to_matrix/ (Sharma, 2021)

Detro(Jan, 2021). Remove Stop Words from Text in DataFrame Column. https://www.datasnips.com/58/remove-stop-words-from-text-in-dataframe-column/ (Detro, 2021)

stackoverflow.com.IOPub data rate exceeded in Jupyter notebook. https://stackoverflow.com/questions/43288550/iopub-data-rate-exceeded-in-jupyter-notebook-when-viewing-image

Hamish(2018). Using Keras OOV Tokens. https://www.kaggle.com/code/hamishdickson/using-keras-oov-tokens/notebook (Hamish, 2018)

TensorFlow Authors(2020). Preparing text to use with TensorFlow models https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c02_nlp_padding.ipynb#scrollTo=RX9Yx50TUies (Authors, 2020)

Keras documentaion. Layer activation functions https://keras.io/api/layers/activations/ (Keras documentaion)

Newdev(2021). What is the difference between sparse_categorical_crossentropy and categorical_crossentropy? https://newbedev.com/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-crossentropy (Newdev2021)

## In-Line Sources
Jason Brownlee (Dec, 2019). TensorFlow 2 Tutorial: Get Started in Deep Learning With tf.keras https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/#:~:text=TensorFlow%20is%20the%20premier%20open,Keras%20to%20the%20TensorFlow%20project.

Ali Hamza (Jan, 2019). Effectively Pre-processing the Text Data Part 1: Text Cleaning https://towardsdatascience.com/effectively-pre-processing-the-text-data-part-1-text-cleaning-9ecae119cb3e

Kristin H. Huseby(June, 2020). Word embeddings, what are they really? https://towardsdatascience.com/word-embeddings-what-are-they-really-f106e1ff0874#:~:text=With%20word%20embeddings%20we%20assign,with%20the%20most%20useful%20results. (Huseby, 2020)

Jeffrey Pennington, Richard Socher, Christopher D. Manning. Computer Science Department, Stanford University. GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/pubs/glove.pdf (Pennington, Socher, Manning)

Jason Brownlee(Oct, 2017). How to Use Word Embedding Layers for Deep Learning with Keras. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

Caner(Apr, 2020). Padding for NLP https://medium.com/@canerkilinc/padding-for-nlp-7dd8598c916a (Caner, 2020)

Aravindpai Pai (May, 2020). What is Tokenization in NLP? Here’s All You Need To Know https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/ (Pai,2020)

Festus Elleh(2022). Cohort Recorded Event. https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=cedbd86a-2543-4d9d-9b0e-aec4011a606d (Elleh, 2022)

TensorFlow Authors(2020). Preparing text to use with TensorFlow models https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l09c02_nlp_padding.ipynb#scrollTo=RX9Yx50TUies (Authors, 2020)

Dwarampudi Mahidhar Reddy @ N V Subba Reddy(Mar, 2019). EFFECTS OF PADDING ON LSTMS AND CNNS https://arxiv.org/pdf/1903.07288.pdf (Reddy, 2019)

Tensorflow Core Guide. Masking and padding with Keras https://www.tensorflow.org/guide/keras/masking_and_padding (Masking and padding with Keras)

Keras. Softmax layer https://keras.io/api/layers/activation_layers/softmax/ (Keras, Softmax layer)

I2 tutorials (Sep, 2019). Home / Deep Learning Interview questions and answers / Explain about Adam Optimization Function? https://www.i2tutorials.com/explain-about-adam-optimization-function/ (I2 tutorials, 2019)

Goutham Dommaraju, (May, 2020). Keras’ Accuracy Metrics https://towardsdatascience.com/keras-accuracy-metrics-8572eb479ec7#:~:text=If%20%281%29%20and%20%282%29%20concur%2C%20attribute%20the%20logical,to%20the%20actual%20value%2C%20it%20is%20considered%20accurate.

Jason Brownlee (Dec, 2018). Use Early Stopping to Halt the Training of Neural Networks At the Right Time https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/ (Brownlee, 2018)
