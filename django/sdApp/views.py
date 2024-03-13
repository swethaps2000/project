from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json

# Load the model
model3 = load_model("sdApp\model\model3.h5")
tokenizer_obj = Tokenizer()
max_length = 25


def hello(request):
    if request.method == 'POST':
        # sentence = request.POST.get('sentence')  
        # print("receieved sentence",sentence)
        data = json.loads(request.body)
        sentence = data.get('sentence', '')
        print(sentence)
    return HttpResponse("Heelo")

def index(request):
    return render(request, 'home.html')


def predict_sarcasm(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        sentence = data.get('sentence', '')
        print("sentence",sentence)
        if sentence:
            prediction = predict_sarcasm3(sentence)
            print("hey",prediction)
            return JsonResponse({'prediction': prediction})
        else:
            return JsonResponse({'error': 'Input sentence is empty'})
    else:
        return JsonResponse({'error': 'Invalid request'})


def predict_sarcasm3(sentence):
    try:
    
        test_lines = CleanTokenize(sentence)
        test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
        test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')

        # Predict
        pred = model3.predict(test_review_pad)
        pred *= 100
        if pred[0][0] >= 50:
            return "It's a sarcasm!"
        else:
            return "It's not a sarcasm."
    except Exception as e:
        return f"Prediction error: {str(e)}"


def clean_text(text):
    text = text.lower()

    pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x: x[0] != '@', text.split()))
    emoji = re.compile("["
                       u"\U0001F600-\U0001FFFF"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text


def CleanTokenize(sentence):
    head_lines = list()
    line = sentence
    line = clean_text(line)
    # tokenize the text
    tokens = word_tokenize(line)
    # remove punctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove non-alphabetic characters
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    # remove stop words
    words = [w for w in words if not w in stop_words]
    head_lines.append(words)
    return head_lines
