NLP | Sentiment Analysis of Company Reviews
Sentiment Analysis
Source: [surveysensum.com | Sentiment Analysis: A Thorough Guide For The Data Geek]
?? Objective¶
The objective for this notebook is to build a baseline model which is capable of predicting the sentiment of company reviews left by customers for the Sentiment Analysis - Company Reviews Competition.

?? So what exactly is Sentiment Analysis?
According to ChatGPT:

Sentiment analysis, also known as opinion mining, is the process of analyzing and identifying the sentiment, attitude, or emotion expressed in a piece of text, such as a review or a social media post.

It is commonly used in various industries, such as marketing, customer service, and politics, to understand people's opinions, preferences, and behavior. Sentiment analysis uses natural language processing (NLP) techniques and algorithms to analyze and classify text into different categories, such as positive, negative, or neutral.The process involves pre-processing the text, such as tokenization and stemming, to convert it into a format that can be analyzed by the algorithms.

Sentiment analysis models can be trained on a dataset of labeled data, which contains examples of text with their corresponding sentiment labels. Alternatively, models can be trained using unsupervised learning techniques, which use clustering and other methods to classify the text without labeled data.

Sentiment analysis can be performed on various types of text, such as social media posts, product reviews, news articles, and customer feedback. Overall, sentiment analysis is a useful tool for understanding people's opinions and attitudes, and can help businesses make data-driven decisions based on customer feedback.

For more information on Sentiment Analysis, see the following links:

MonkeyLearn | Sentiment Analysis: A Definitive Guide
Thematic | Sentiment Analysis: Comprehensive Beginners Guide
?? Dataset
The dataset for this competition (both train and test) consists of 100,000 reviews collected from Trustpilot and spans over 40 different companies.

Find this competitions dataset here: Sentiment Analysis - Company Reviews Dataset



Table of contents
1 | Dataset Exploration

Load CSV Files
View Random Selected Samples
View Train Rating Distribution
Inspect Review Lengths & Tokens
View Review Lengths & Review Token Count Histograms
2 | Data Preprocessing

Label Encode Ratings
Create Train/Validation Split
View New Train & Validation Labels Distribution
3 | Build Input Data Pipeline with tf.data API

Define Text Preprocessor
Generate Input Data Pipelines
4 | Baseline Model: Universal Sentence Encoder Model

TensorFlow Hub
Get Universal Sentence Encoder
Build Model
5 | Train Baseline Model

Define Callbacks and Metrics for Model Training
Compile & Train Model
6 | Model Performance Evaluation

View Model Histories
Plot Confusion Matrix
Generate Classification Reports
Record Classification Metrics
7 | Generate Submission

Preprocess Test Reviews
Generate Test Predictions
Generate Submission.csv
Conclusion



import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, mean_absolute_error
from scikitplot.metrics import plot_roc
class CFG:
    SEED = 768
    BATCH_SIZE = 32
    EPOCHS = 10

1 | Dataset Exploration
# Define paths
DATASET_PATH = "/kaggle/input/sentiment-analysis-company-reviews/"
TRAIN_CSV = '/kaggle/input/sentiment-analysis-company-reviews/train.csv'
TEST_CSV = '/kaggle/input/sentiment-analysis-company-reviews/test.csv'
SAMPLE_SUB_CSV = '/kaggle/input/sentiment-analysis-company-reviews/sample_submission.csv'
Load CSV Files
# Load the csv files
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
submission_df = pd.read_csv(SAMPLE_SUB_CSV) 
# Generate summary of the training set
train_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60000 entries, 0 to 59999
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Id      60000 non-null  int64 
 1   Review  60000 non-null  object
 2   Rating  60000 non-null  int64 
dtypes: int64(2), object(1)
memory usage: 1.4+ MB
# View first 5 training samples
train_df.head(5)
Id	Review	Rating
0	0	Very good value and a great tv very happy and ...	5
1	1	After 6 month still can't access my account	3
2	2	I couldn't make an official review on a produc...	1
3	3	Fantastic! Extremely easy to use website, fant...	5
4	4	So far annoyed as hell with this bt monthly pa...	1

View Random Selected Samples
def view_samples(df, count=5):
    idx = random.sample(train_df.index.to_list(), count)
    print('=========================================\n')
    for _ in idx:
        print(f'id:\t{df.Id[_]}\n')
        print(f'Review:\n{df.Review[_]}\n')
        print(f'Rating:\n{df.Rating[_]}')
        print('=========================================\n')
# View 5 randomly selected samples
view_samples(train_df, count=5)
=========================================

id:	29867

Review:
Been a loyal customer for years as has family members. Went to shop today to arrange an upgrade on my IPhone. Was shown deals and agreed an upgrade deal. Shop staff were lovely. Happened to look online to see Carphone warehouse have a much better deal for the same phone/contract (O2) online, in fact £16 a month cheaper so nearly £400 saving over 2 year contract. Went to web chat and was told online and shop are different and cannot change. Said as existing customer and upgrade no cooling off period. Rang shop and they said could have matched deal if knew about it but they don’t and to ask for upgrade to be reset. Web advisors refused. How is it right that new customers have more rights than existing and even in the same company there is no ability to allow you to change to a more reasonable contract after no more than 2 hours after you have been to the shop? Where is the support for customer loyalty here? I feel totally and utterly upset and let down by Carphone warehouse.

Rating:
1
=========================================

id:	33311

Review:
Great customer service. When I forgot to claim a free gift I reached out and your team assisted me with getting that free gift. Thanks been using PB for years now.

Rating:
5
=========================================

id:	38171

Review:
Terrible, terrible company. I haven't been able to sign into my account on any device for weeks and they only offer online help that requires me to sign in (I can't). No phone number, no email to contact about the problem and no way to cancel the payments that continue to come out of my bank account except getting my bank to block Now TV and then disputing the payments.

Rating:
1
=========================================

id:	13011

Review:
Awful , incompetent, was supposed to broadband and phone installed on 27th June, when the Hub arrived it stated 29th June, no mention of the 2 day delay to me at all, needless to say 29th came and went , i had to phone them they blamed Open reach. Date was pushed back and back with the final date given 2nd August, I cancelled this on the 11th July, due to the fact they could not provide the service i signed up for. Then last week a rude operative rang me to tell me my internet was up and running, i informed the person that this service was no longer required and that i cancelled it .. so what did Now do ,helped themselves to £35.00 out my bank , so i rang them yet again and was passed from pillar to post until the last person i spoke to said that this had been cancelled and they had no idea why it hadnt went through! he told me that they was sort out a refund...not happend! then on checking my account today to see if it had actually been cancelled , there is now another £35.00 payment scheduled for this month! This is awful in itself but the original service i signed up for was £22.00 per month ! I am contacting their governing body to ask them to investigate it and i have contacted their complaints department. I would advise anyone to stay clear of these people they are so incompetent .

Rating:
1
=========================================

id:	27567

Review:
Ordering was easy, battery turned up the following day as promised.
Many thanks
Steve W

Rating:
5
=========================================


View Train Rating Distribution
# View Train Rating Distribution
plt.figure(figsize=(15, 8))
plt.title('Train Rating Distribution', fontsize=20)

train_distribution = train_df['Rating'].value_counts().sort_values()
sns.barplot(x=list(train_distribution.keys()),
            y=train_distribution.values);

Observation
We observe that the dataset is severly imbalanced with review with ratings of 1 and 5 make up the majority of the dataset's reviews. Techniques such as undersampling, oversampling or weighted training should be considered. However, this notebook will only focus on the baseline model and will not cover the implementation of these techniques. It should be noted that this behavior in the data is most likely representative of customer behavior (customers would rather give a rating of 1 or 5 and ratings in between siginify mixed opinions).

Inspect Review Lengths & Tokens
# Get the lengths of each review
train_df['review_length'] = [len(_) for _ in train_df.Review]

# Get the number of tokens per review 
train_df['token_count'] = [len(_.split()) for _ in train_df.Review]
# View first 5 samples 
train_df.head(5)
Id	Review	Rating	review_length	token_count
0	0	Very good value and a great tv very happy and ...	5	89	18
1	1	After 6 month still can't access my account	3	43	8
2	2	I couldn't make an official review on a produc...	1	496	92
3	3	Fantastic! Extremely easy to use website, fant...	5	197	32
4	4	So far annoyed as hell with this bt monthly pa...	1	222	49
# Inspect Review Length Stats
print('Review Length Description')
print('==================================')
print(train_df['review_length'].describe())
print('==================================')
Review Length Description
==================================
count    60000.000000
mean       309.070083
std        423.772492
min         31.000000
25%         82.000000
50%        164.000000
75%        358.000000
max       7794.000000
Name: review_length, dtype: float64
==================================
# Inspect Token Count Stats
print('Token Count Description')
print('==================================')
print(train_df['token_count'].describe())
print('==================================')
Token Count Description
==================================
count    60000.00000
mean        56.56325
std         79.35289
min          1.00000
25%         14.00000
50%         29.00000
75%         66.00000
max       1439.00000
Name: token_count, dtype: float64
==================================
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 18))

# Set the spacing between subplots
fig.tight_layout(pad=6.0)

# Plot Range of Review Lengths per Rating
ax1.set_title('Review Lengths per Rating', fontsize=20)
sns.boxplot(data=train_df, y='review_length', x='Rating',
            ax=ax1)

# Plot Range of Token Counts per Rating
ax2.set_title('Token Counts per Rating', fontsize=20)
sns.boxplot(data=train_df, y='token_count', x='Rating',
            ax=ax2);

Observation
We observe that the length of the reviews increase the more unsatisfied the customers are with the companies. The same observation can be made for the number of tokens per review. The reason for this may be that customers tend to explain or describe their opinions/experiences in great detail the more unsatisfied they are with the companies. This reason may also explain why reviews with higher ratings are generally shorter with less tokens present as this signifies satification amongst customers.

View Review Lengths & Review Token Count Histograms
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))

# Set the spacing between subplots
fig.tight_layout(pad=6.0)

# Generate Train Rating Histogram
ax1.set_title('Train Review Length Histogram', fontsize=20)
sns.histplot(data=train_df, x='review_length', bins=50,
            ax=ax1)

# Generate Train Token Count Histogram
ax2.set_title('Train Token Count Histogram', fontsize=20)
sns.histplot(data=train_df, x='token_count', bins=50,
            ax=ax2);

Observation
We observe that the majority of review lengths are under a length of ~1000. We also observe that the review token counts are generally under ~300 tokens. These factors should be considered when selecting the number of tokens to be used in a model. Selecting the number of tokens to be used in model via percentiles may prove to be helpful. However, this will not be covered in this notebook.
?? Back To Top

2 | Data Preprocessing

Label Encode Ratings
We need to label encode the ratings since all ratings fall in the range between 1 and 5 inclusively. To achieve this we simply shift the ratings by subtracting 1 from each rating (e.g. 5 -> 4). We do this in order to simplify the one-hot encoding process at a later stage.

# Label encode ratings
train_df["rating_encoded"] = train_df['Rating'] - 1

Create Train/Validation Split
# Get indices for train and validation splits
train_idx, val_idx, _, _ = train_test_split(
    train_df.index, train_df.Rating, 
    test_size=0.2, stratify=train_df.Rating,
    random_state=CFG.SEED
)
# Get new training and validation data
train_new_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

# View shapes
train_new_df.shape, val_df.shape
((48000, 6), (12000, 6))
# View new train dataframe
train_new_df
Id	Review	Rating	review_length	token_count	rating_encoded
0	7372	Straight forward purchase from trusted supplie...	5	80	12	4
1	36260	Website is super smooth and easy to navigate w...	5	118	21	4
2	33497	Apparently it was posted twice, maybe it will ...	1	150	26	0
3	36775	Order received the day after order. Memory upg...	5	94	15	4
4	49630	Its great, in a contract with them for 30GB fo...	5	162	27	4
...	...	...	...	...	...	...
47995	40748	I have made 2 purchases with PCSPECIALIST in t...	5	91	17	4
47996	33270	Matress is amazing- thank you\nYet again deliv...	2	402	78	1
47997	16802	Gtech have a great product I always buy from t...	5	134	25	4
47998	37531	Quick delivery, competitive price	5	33	4	4
47999	13469	Signed up to get broadband installed originall...	1	697	133	0
48000 rows × 6 columns


View Train & Validation Rating Distributions
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10))

# Set the spacing between subplots
fig.tight_layout(pad=6.0)

# Plot New Train Ratings Distribution
ax1.set_title('New Train Ratings Distribution', fontsize=20)
train_new_distribution = train_new_df['Rating'].value_counts().sort_values()
sns.barplot(x=train_new_distribution.values,
            y=list(train_new_distribution.keys()),
            orient="h",
            ax=ax1)

# Plot Validation Ratings Distribution
ax2.set_title('Validation Ratings Distribution', fontsize=20)
val_distribution = val_df['Rating'].value_counts().sort_values()
sns.barplot(x=val_distribution.values,
            y=list(val_distribution.keys()),
            orient="h",
            ax=ax2);

?? Back To Top

3 | Build Input Data Pipeline with tf.data API

In this notebook we'll use the tf.data API to build input data pipelines for training a model and conducting model inference. In order to achieve this, we'll preprocess the reviews by removing any artifacts in the texts such as emojis, non-ascii characters and replacing numbers with another character. The preprocessed texts will be used to construct the pipelines along with the one-hot encoded ratings.

UCF101
Source: [Medium | How to Reduce Training Time for a Deep Learning Model using tf.data]


For more information on the tf.data API and loading data from generator, follow these links:

tf.data: Build TensorFlow input pipelines
Better performance with the tf.data API
Using generators with tf.data API

Define Text Preprocessor
def text_preprocessor(text):
    
    # -----------------------------------------------------
    # Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    # -----------------------------------------------------
    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+", flags=re.UNICODE)
    digit_pattern = re.compile('[0-9]', flags=re.UNICODE)
    
    # -----------------------------------------------------
    # Source: https://stackoverflow.com/questions/21932615/regular-expression-for-remove-link
    link_pattern = re.compile('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', flags=re.UNICODE)
    # -----------------------------------------------------
    
    # Remove emojis
    preprocessed_text = emoji_pattern.sub(r'', text)
    # Remoce non-ascii characters
    preprocessed_text = non_ascii_pattern.sub(r'', preprocessed_text)
    # Replace numbers with '@' sign
    preprocessed_text = digit_pattern.sub(r'#', preprocessed_text)
    # Remove web links 
    preprocessed_text = link_pattern.sub(r'', preprocessed_text)
    
    return preprocessed_text

Generate Input Data Pipelines
def encode_labels(labels, label_depth=5):
    return tf.one_hot(labels, depth=label_depth).numpy()

def create_pipeline(df, preprocessor, batch_size=32, shuffle=False, cache=None, prefetch=False):
    '''
    Generates an input pipeline using the tf.data API given a Pandas DataFrame and image loading function.
    
    @params
        - df: (pd.DataFrame) -> DataFrame containing texts and labels
        - preprocessor (function) -> preprocessor used to preprocess texts
        - batch_size: (int) -> size for batched (default=32) 
        - shuffle: (bool) -> condition for data shuffling, data is shuffled when True (default=False)
        - cache: (str) -> cache path for caching data, data is not cached when None (default=None)
        - prefetch: (bool) -> condition for prefeching data, data is prefetched when True (default=False)
        
    @returns
        - dataset: (tf.data.Dataset) -> dataset input pipeline used to train a TensorFlow model
    '''
    # Get image paths and labels from DataFrame
    reviews = df['Review'].apply(preprocessor).to_numpy().astype(str)
    ratings = encode_labels(df['rating_encoded'].to_numpy().astype(np.float32))
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Create dataset with raw data from DataFrame
    ds = tf.data.Dataset.from_tensor_slices((reviews, ratings))
    
    # Apply shuffling based on condition
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
        
    # Apply batching
    ds = ds.batch(batch_size)
    
    # Apply caching based on condition
    # Note: Use cache in memory (cache='') if the data is small enough to fit in memory!!!
    if cache != None:
        ds = ds.cache(cache)
    
    # Apply prefetching based on condition
    # Note: This will result in memory trade-offs
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    # Return the dataset
    return ds
# Create train input data pipeline
train_ds = create_pipeline(
    train_new_df, text_preprocessor, 
    batch_size=CFG.BATCH_SIZE, 
    shuffle=False, prefetch=True
)

# Create validation input data pipeline
val_ds = create_pipeline(
    val_df, text_preprocessor,
    batch_size=CFG.BATCH_SIZE, 
    shuffle=False, prefetch=False
)
# View string representation of datasets
print('========================================')
print('Train Input Data Pipeline:\n\n', train_ds)
print('========================================')
print('Validation Input Data Pipeline:\n\n', val_ds)
print('========================================')
========================================
Train Input Data Pipeline:

 <PrefetchDataset shapes: ((None,), (None, 5)), types: (tf.string, tf.float32)>
========================================
Validation Input Data Pipeline:

 <BatchDataset shapes: ((None,), (None, 5)), types: (tf.string, tf.float32)>
========================================
?? Back To Top

4 | Baseline Model: Universal Sentence Encoder Model

The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks. For this baseline model we'll make use of Universal Sentence Encoder (USE) to generate embeddings which are representative of the review texts.

UCF101
Source: [Amit Chaudhary | Universal Sentence Encoder Visually Explained]
Reasons for using Universal Sentence Encoder:

Minimal hardware requirements for generating embeddings with USE
Low inference rate
Light-weight memory consumptions
Drawbacks of using Universal Sentence Encoder:

Embedding representations become less accurate as text lengths increases.
Although USE has a low inference rate, its accuracy falls short when compared to language models such as BERT & DeBERTa-v3


For more information regarding the Universal Sentence Encoder, follow these links:

An Introduction to Transfer Learning
A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning
Amit Chaudhary | Universal Sentence Encoder Visually Explained

TensorFlow Hub
TensorFlow Hub is a repository of trained machine learning models ready for fine-tuning and deployable anywhere. TensorFlow Hub enables us to reuse trained models like BERT and Faster R-CNN with just a few lines of code. In this section we'll get the USE model from TensorFlow Hub.

For more information on TensorFlow Hub or if you would like to access the other models in PyTorch/JAX, check out the following links:

TensorFlow Hub
HuggingFace??
# Here's a function to get any model/preprocessor from tensorflow hub
def get_tfhub_model(model_link, model_name, model_trainable=False):
    return hub.KerasLayer(model_link,
                          trainable=model_trainable,
                          name=model_name)

Get Universal Sentence Encoder
# Get Universal Sentence Encoder here
# -----------------------------------
# Note: We'll use the version from Kaggle's Models page instead.
#       Check it out here: 
#       (https://www.kaggle.com/models/google/universal-sentence-encoder)
# -----------------------------------
encoder_link = 'https://kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2'
# encoder_link = 'https://tfhub.dev/google/universal-sentence-encoder/4'

encoder_name = 'universal_sentence_encoder'
encoder_trainable=False # set trainable to False for inference-only 

encoder = get_tfhub_model(encoder_link, encoder_name, model_trainable=encoder_trainable)

Build Model
def build_baseline_model(num_classes=5):
    # Define kernel initializer & input layer
    initializer = tf.keras.initializers.HeNormal(seed=CFG.SEED)
    review_input = layers.Input(shape=[], dtype=tf.string, name='review_text_input')
    
    # Generate Embeddings
    review_embedding = encoder(review_input)
    
    # Feed Embeddings to a Bidirectional LSTM
    expand_layer = layers.Lambda(lambda embed: tf.expand_dims(embed, axis=1))(review_embedding)
    bi_lstm = layers.Bidirectional(layers.LSTM(128, kernel_initializer=initializer), 
                                   name='bidirection_lstm')(expand_layer)
    
    # Feed LSTM output to classification head
    dropout_layer = layers.Dropout(0.25)(bi_lstm)
    dense_layer = layers.Dense(64, activation='relu', kernel_initializer=initializer)(dropout_layer)
    output_layer = layers.Dense(num_classes, activation='softmax', 
                                kernel_initializer=initializer, 
                                name='output_layer')(dense_layer)
    
    return tf.keras.Model(inputs=[review_input], 
                          outputs=[output_layer], 
                          name='use_model')
# Build model
model = build_baseline_model()

# View summary of model
model.summary()
Model: "use_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
review_text_input (InputLaye [(None,)]                 0         
_________________________________________________________________
universal_sentence_encoder ( (None, 512)               256797824 
_________________________________________________________________
lambda (Lambda)              (None, 1, 512)            0         
_________________________________________________________________
bidirection_lstm (Bidirectio (None, 256)               656384    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                16448     
_________________________________________________________________
output_layer (Dense)         (None, 5)                 325       
=================================================================
Total params: 257,470,981
Trainable params: 673,157
Non-trainable params: 256,797,824
_________________________________________________________________
# Explore model visually
plot_model(
    model, dpi=60,
    show_shapes=True,
    expand_nested=True
)

?? Back To Top

5 | Train Baseline Model

To train this model we'll use Categorical Crossentropy as the loss function since this notebook approaches the problem at hand as a classification problem for multiple labels. As for the optimizer, we'll use the Adam optimizer with 0.001 as the (default) learning rate.

To prevent the occurance of overfitting during training we'll have to make use of TensorFlow's Callback API to implement the EarlyStopping & ReduceLROnPlateau callbacks. The only metrics we'll track during the training of the model will be the loss and accuracy metrics.

See the following for more information:

Categorical Crossentropy Loss Function:
Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names
TensorFlow Categorical Crossentropy Loss Implementation
Adam Optimizer:
Academic Paper | Adam: A Method for Stochastic Optimization
TensorFlow Adam Implementation
TensorFlow Callback API:
EarlyStopping Implementation
ReduceLROnPlateau Implementation
TensorFlow Metrics:
TensorFlow Metrics Overview
def train_model(model, num_epochs, callbacks_list, tf_train_data, 
                tf_valid_data=None, shuffling=False):
    '''
        Trains a TensorFlow model and returns a dict object containing the model metrics history data. 
        
        @params
        - model: (tf.keras.model) -> model to be trained 
        - num_epochs: (int) -> number of epochs to train the model
        - callbacks_list: (list) -> list containing callback fuctions for model
        - tf_train_data: (tf.data.Dataset) -> dataset for model to be train on 
        - tf_valid_data: (tf.data.Dataset) -> dataset for model to be validated on (default=None)
        - shuffling: (bool) -> condition for data shuffling, data is shuffled when True (default=False)
        
        @returns
        - model_history: (dict) -> dictionary containing loss and metrics values tracked during training
    '''
    
    model_history = {}
    
    if tf_valid_data != None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  validation_data=tf_valid_data,
                                  validation_steps=int(len(tf_valid_data)),
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)
        
    if tf_valid_data == None:
        model_history = model.fit(tf_train_data,
                                  epochs=num_epochs,
                                  callbacks=callbacks_list,
                                  shuffle=shuffling)
    return model_history

Define Callbacks and Metrics for Model Training
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.1,
    verbose=1)

CALLBACKS = [early_stopping_callback, reduce_lr_callback]
METRICS = ['accuracy']

Compile & Train Model
tf.random.set_seed(CFG.SEED)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=METRICS
)

print(f'Training {model.name}.')
print(f'Train on {len(train_new_df)} samples, validate on {len(val_df)} samples.')
print('----------------------------------')

model_history = train_model(
    model, CFG.EPOCHS, CALLBACKS, 
    train_ds, val_ds,
    shuffling=False
)
Training use_model.
Train on 48000 samples, validate on 12000 samples.
----------------------------------
Epoch 1/10
1500/1500 [==============================] - 56s 33ms/step - loss: 0.7271 - accuracy: 0.8549 - val_loss: 0.6951 - val_accuracy: 0.8652
Epoch 2/10
1500/1500 [==============================] - 46s 31ms/step - loss: 0.6934 - accuracy: 0.8652 - val_loss: 0.6896 - val_accuracy: 0.8672
Epoch 3/10
1500/1500 [==============================] - 46s 31ms/step - loss: 0.6850 - accuracy: 0.8680 - val_loss: 0.6870 - val_accuracy: 0.8679
Epoch 4/10
1500/1500 [==============================] - 46s 31ms/step - loss: 0.6781 - accuracy: 0.8705 - val_loss: 0.6854 - val_accuracy: 0.8685
Epoch 5/10
1500/1500 [==============================] - 44s 30ms/step - loss: 0.6715 - accuracy: 0.8724 - val_loss: 0.6840 - val_accuracy: 0.8688
Epoch 6/10
1500/1500 [==============================] - 44s 29ms/step - loss: 0.6651 - accuracy: 0.8749 - val_loss: 0.6835 - val_accuracy: 0.8683
Epoch 7/10
1500/1500 [==============================] - 44s 29ms/step - loss: 0.6582 - accuracy: 0.8772 - val_loss: 0.6849 - val_accuracy: 0.8689
Epoch 8/10
1500/1500 [==============================] - 44s 29ms/step - loss: 0.6507 - accuracy: 0.8808 - val_loss: 0.6848 - val_accuracy: 0.8684

Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 9/10
1500/1500 [==============================] - 44s 29ms/step - loss: 0.6354 - accuracy: 0.8859 - val_loss: 0.6822 - val_accuracy: 0.8693
Epoch 10/10
1500/1500 [==============================] - 44s 30ms/step - loss: 0.6330 - accuracy: 0.8872 - val_loss: 0.6825 - val_accuracy: 0.8696
# Evaluate the model
model_evaluation = model.evaluate(val_ds)
375/375 [==============================] - 8s 20ms/step - loss: 0.6825 - accuracy: 0.8696
# Generate model probabilities and associated predictions
train_probabilities = model.predict(train_ds, verbose=1)
train_predictions = tf.argmax(train_probabilities, axis=1)
1500/1500 [==============================] - 32s 20ms/step
# Generate model probabilities and associated predictions
val_probabilities = model.predict(val_ds, verbose=1)
val_predictions = tf.argmax(val_probabilities, axis=1)
375/375 [==============================] - 7s 20ms/step
?? Back To Top

6 | Model Performance Evaluation

Now that the model has trained on the data we need to inspect how well it performs on the validation data. In order to conduct this inspection we need to evaluate the performance of the model on the validation data and record evaluation metrics. Since the approach for this problem is a multi classification problem we'll make use of some well known classification metrics. Hence, we'll make use of the Scikit Learn library to inspect the model. We'll also use the following to inspect the model:

Classification Report
Accuracy Score
Precision
Recall
F1-score
Matthews Correlation Coefficient

Plot Model Training History
def plot_training_curves(history):
    
    loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])

    accuracy = np.array(history.history['accuracy'])
    val_accuracy = np.array(history.history['val_accuracy'])

    epochs = range(len(history.history['loss']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot loss
    ax1.plot(epochs, loss, label='training_loss', marker='o')
    ax1.plot(epochs, val_loss, label='val_loss', marker='o')
    
    ax1.fill_between(epochs, loss, val_loss, where=(loss > val_loss), color='C0', alpha=0.3, interpolate=True)
    ax1.fill_between(epochs, loss, val_loss, where=(loss < val_loss), color='C1', alpha=0.3, interpolate=True)

    ax1.set_title('Loss (Lower Means Better)', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, accuracy, label='training_accuracy', marker='o')
    ax2.plot(epochs, val_accuracy, label='val_accuracy', marker='o')
    
    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy > val_accuracy), color='C0', alpha=0.3, interpolate=True)
    ax2.fill_between(epochs, accuracy, val_accuracy, where=(accuracy < val_accuracy), color='C1', alpha=0.3, interpolate=True)

    ax2.set_title('Accuracy (Higher Means Better)', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.legend();
# plot model training history 
plot_training_curves(model_history)

Observation
We observe that overfitting may have occured during the first few expochs. We also observe that the model reached a plateau on the validation loss.

Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes='auto', figsize=(10, 10), text_size=12): 
    # Generate confusion matrix 
    cm = confusion_matrix(y_true, y_pred)
    
    # Set plot size
    plt.figure(figsize=figsize)

    # Create confusion matrix heatmap
    disp = sns.heatmap(
        cm, annot=True, cmap='Greens',
        annot_kws={"size": text_size}, fmt='g',
        linewidths=1, linecolor='black', clip_on=False,
        xticklabels=classes, yticklabels=classes)
    
    # Set title and axis labels
    disp.set_title('Confusion Matrix', fontsize=24)
    disp.set_xlabel('Predicted Label', fontsize=20) 
    disp.set_ylabel('True Label', fontsize=20)
    plt.yticks(rotation=0) 

    # Plot confusion matrix
    plt.show()
    
    return
plot_confusion_matrix(
    val_df.Rating - 1, 
    val_predictions, 
    figsize=(10, 10))

Observation
The model is able to classify the majority classes. However, the characteristics of a severly imbalanced dataset is present as the model struggles with predicting the minority classes.

Generate Classification Report
print(classification_report(val_df.Rating - 1, val_predictions))
              precision    recall  f1-score   support

           0       0.86      0.96      0.91      3732
           1       0.17      0.01      0.02       326
           2       0.25      0.12      0.16       336
           3       0.38      0.12      0.19       670
           4       0.91      0.97      0.94      6936

    accuracy                           0.87     12000
   macro avg       0.51      0.44      0.44     12000
weighted avg       0.82      0.87      0.84     12000


Record Classification Metrics
def generate_preformance_scores(y_true, y_pred, y_probabilities):
    
    model_accuracy = accuracy_score(y_true, y_pred)
    top_2_accuracy = top_k_accuracy_score(y_true, y_probabilities, k=2)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, 
                                                                                 y_pred, 
                                                                                 average="weighted")
    model_matthews_corrcoef = matthews_corrcoef(y_true, y_pred)
    
    print('=============================================')
    print(f'\nPerformance Metrics:\n')
    print('=============================================')
    print(f'accuracy_score:\t\t{model_accuracy:.5f}\n')
    print('_____________________________________________')
    print(f'top_2_accuracy_score:\t{top_2_accuracy:.5f}\n')
    print('_____________________________________________')
    print(f'precision_score:\t{model_precision:.5f}\n')
    print('_____________________________________________')
    print(f'recall_score:\t\t{model_recall:.5f}\n')
    print('_____________________________________________')
    print(f'f1_score:\t\t{model_f1:.5f}\n')
    print('_____________________________________________')
    print(f'matthews_corrcoef:\t{model_matthews_corrcoef:.5f}\n')
    print('=============================================')
    
    preformance_scores = {
        'accuracy_score': model_accuracy,
        'top_2_accuracy_score': top_2_accuracy,
        'precision_score': model_precision,
        'recall_score': model_recall,
        'f1_score': model_f1,
        'matthews_corrcoef': model_matthews_corrcoef
    }
    
    return preformance_scores
model_performance = generate_preformance_scores(val_df.Rating-1, val_predictions, val_probabilities)
=============================================

Performance Metrics:

=============================================
accuracy_score:		0.86958

_____________________________________________
top_2_accuracy_score:	0.94458

_____________________________________________
precision_score:	0.82286

_____________________________________________
recall_score:		0.86958

_____________________________________________
f1_score:		0.83834

_____________________________________________
matthews_corrcoef:	0.76070

=============================================
# Inspect Competition Metric: Mean-Absolute-Error
print('Competition Metric Score')
print('=========================')
print(f'Train MAE:\t{mean_absolute_error(train_new_df.rating_encoded, train_predictions):.5f}')
print(f'Validation MAE:\t{mean_absolute_error(val_df.rating_encoded, val_predictions):.5f}')
print('=========================')
Competition Metric Score
=========================
Train MAE:	0.18375
Validation MAE:	0.23225
=========================
Observation
The model was able to achive a Matthews Correlation Coefficient of ~0.76 which is decent. A high MCC implies that the model's predictions are statistically of high quality and that the model may generalise to unseen samples. Looking at the Compition Metric, the model seems to score a low validation MAE and we can observe a significant difference between the training and validation MAE scores.

Since the model was able to score a high MCC, we should expect a similar LB MAE score as the validation MAE.
?? Back To Top

7 | Generate Submission

With the model performance evaluation complete, we need to generate the submission file.


Preprocess Test Reviews
def predict(model, test_reviews):
    probabilities = model.predict(test_reviews, verbose=1)
    predictions = tf.argmax(probabilities, axis=1)
    return probabilities, predictions
# Preprocess Test Reviews
test_reviews = test_df['Review'].apply(text_preprocessor)
test_reviews.shape
(40000,)

Generate Test Predictions
# Generate Test Predictions
test_probabilities, test_predictions = predict(model, test_reviews)
1250/1250 [==============================] - 27s 21ms/step

# Use the sample_subission dataframe to create the
# submission csv for the test set predictions
submission_df['Rating'] = test_predictions + 1 # Decode labels 

# View first 5 submission samples 
submission_df.head(5)
Id	Rating
0	60000	1
1	60001	5
2	60002	1
3	60003	5
4	60004	1
# View Test Predictions Ratings Distribution
plt.figure(figsize=(14, 8))
plt.title('Test Prediction Ratings Distribution', fontsize=20)
test_predictions_distribution = submission_df['Rating'].value_counts().sort_values()

sns.barplot(x=test_predictions_distribution.values,
            y=list(test_predictions_distribution.keys()),
            orient="h");

Observation
The test predictions resemble the train ratings distribution. However, we observe that the minority classes are overshadowed by the majority classes. This behavior should be addressed in future experiments.

Generate Submission.csv
# Create submission csv
submission_df.to_csv('submission.csv', index=False)
?? Back To Top

Conclusion

In this notebook we built a baseline model to predict the sentiment of company reviews left by customers. We achieved this by using the Universal Sentence Encoder to generate text embeddings which where representative of the review texts, and fed these embeddings to a classification head. We achieved a Matthews Correlation Coefficient of ~0.76, which implies that the model's predictions are statistically of high quality and that the model will generalise to unseen samples.

However, when looking at the model's confusion matrix and classification report we observe the characteristics of a severly imbalanced dataset. The model is unable to correctly predict review ratings which are between 1 and 5. Also, the classification approach might be adding to this bad model behavior as the loss function does not aim to minimize the competition metric (Mean Absolute Error). Therefore, it is recommended that an ordinal text regression approach should be followed as this approach will be focuse on minimizing the competition's metric and may achieve better results.



Thank you for taking the time to check out my notebook and I hope you found this insightful!