# 🔤 NLP Projects - Multi-Technique Text Processing

A **comprehensive NLP portfolio** demonstrating classic and modern text processing techniques including sentiment analysis, text classification, word embeddings, and transformer-based models for various natural language understanding tasks.

## 🎯 Overview

This project showcases:
- ✅ Sentiment analysis (positive, negative, neutral)
- ✅ Text classification (spam, reviews, categories)
- ✅ Word embeddings (Word2Vec, GloVe, FastText)
- ✅ Named entity recognition (NER)
- ✅ Transformer models (BERT, DistilBERT)
- ✅ Text preprocessing pipelines
- ✅ Language models and chatbots

## 🏗️ Architecture

### NLP Processing Pipeline
```
Raw Text → Preprocessing → Feature Extraction → Model → Output
            ├─ Tokenization      ├─ TF-IDF         ├─ Classifier
            ├─ Lowercasing       ├─ Word Embeddings├─ Sentiment
            ├─ Stop word removal ├─ Transformers   ├─ NER
            └─ Lemmatization     └─ Contextual     └─ Generation
```

### Tech Stack
| Component | Technology |
|-----------|-----------|
| **Core NLP** | NLTK, spaCy, TextBlob |
| **Deep Learning** | TensorFlow, PyTorch |
| **Embeddings** | Word2Vec, GloVe, FastText |
| **Transformers** | Hugging Face, BERT |
| **Tools** | VADER, Pattern, Gensim |

## 🧹 Text Preprocessing

### Complete Pipeline

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required resources
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stop words
nltk.download('wordnet')  # Lemmatizer
nltk.download('averaged_perceptron_tagger')  # POS tagger

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Step 3: Remove special characters (keep alphanumeric & spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Step 4: Tokenize
        tokens = word_tokenize(text)
        
        # Step 5: Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Step 6: Lemmatization
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Step 7: Remove short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens

# Example
preprocessor = TextPreprocessor()
raw_text = "I really loved this amazing product! 😊 Check it out: https://example.com"
cleaned = preprocessor.preprocess(raw_text)
print(cleaned)  # ['really', 'loved', 'amazing', 'product']
```

## 😊 Sentiment Analysis

### VADER (Valence Aware Dictionary + Sentiment Reasoner)

```python
from nltk.sentiment import SentimentIntensityAnalyzer

# VADER: Excellent for social media and informal text
sia = SentimentIntensityAnalyzer()

# Examples
texts = [
    "This product is absolutely amazing! 😍",
    "Worst experience ever!!! 😠",
    "It's okay, nothing special.",
    "Not good, but not terrible either."
]

for text in texts:
    scores = sia.polarity_scores(text)
    
    print(f"\nText: {text}")
    print(f"Compound Score: {scores['compound']:.3f}")  # -1 to 1
    print(f"Positive: {scores['pos']:.3f}")
    print(f"Negative: {scores['neg']:.3f}")
    print(f"Neutral: {scores['neu']:.3f}")
    
    # Classification
    if scores['compound'] >= 0.05:
        sentiment = "POSITIVE ✓"
    elif scores['compound'] <= -0.05:
        sentiment = "NEGATIVE ✗"
    else:
        sentiment = "NEUTRAL"
    print(f"Sentiment: {sentiment}")

# Output example:
# Text: This product is absolutely amazing! 😍
# Compound Score: 0.897
# Positive: 0.436
# Negative: 0.000
# Neutral: 0.564
# Sentiment: POSITIVE ✓
```

### Deep Learning Sentiment (BERT-based)

```python
from transformers import pipeline

# Load pretrained sentiment classifier
classifier = pipeline('sentiment-analysis', 
                     model='distilbert-base-uncased-finetuned-sst-2-english')

texts = [
    "This movie is fantastic! Best film of the year.",
    "Absolutely terrible, waste of money.",
    "It was okay, nothing special."
]

for text in texts:
    result = classifier(text)
    print(f"\nText: {text}")
    print(f"Label: {result[0]['label']}")  # POSITIVE or NEGATIVE
    print(f"Score: {result[0]['score']:.3f}")  # Confidence 0-1
```

## 🏷️ Text Classification

### TF-IDF + Logistic Regression (Spam Detection)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Dataset: Email spam vs ham
emails = [
    ("Congratulations! You've won a prize!", True),  # Spam
    ("Your flight confirmation is attached", False),  # Ham
    ("Click here for FREE money!!!", True),  # Spam
    ("Meeting at 3pm in conference room", False),  # Ham
    # ... more examples
]

X = [text for text, _ in emails]
y = [label for _, label in emails]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, 
                           target_names=['Ham', 'Spam']))

# Prediction on new text
new_email = "Limited offer! Buy now and get 50% off!"
new_tfidf = vectorizer.transform([new_email])
prob = classifier.predict_proba(new_tfidf)[0]
print(f"Spam probability: {prob[1]:.1%}")
```

### CNN for Text Classification

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Prepare data
texts = ["great movie", "terrible film", "amazing show", "awful"]
labels = [1, 0, 1, 0]  # Positive/Negative

# Tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10, padding='post')

# CNN for text
model = Sequential([
    # Embedding: Convert integers to word vectors
    Embedding(input_dim=1000, output_dim=128, input_length=10),
    
    # Convolutional filters of size 3, 4, 5
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPool1D(pool_size=2),
    
    # Dense layers
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, labels, epochs=10)

# Prediction
test_seq = tokenizer.texts_to_sequences(["very good"])
test_padded = pad_sequences(test_seq, maxlen=10, padding='post')
pred = model.predict(test_padded)
print(f"Review is positive: {pred[0][0]:.1%}")
```

## 📚 Word Embeddings

### Word2Vec (Skip-gram model)

```python
from gensim.models import Word2Vec

# Training corpus
corpus = [
    "Machine learning is powerful",
    "Deep learning requires data",
    "Neural networks learn patterns",
    "Data science uses statistics"
]

sentences = [text.split() for text in corpus]

# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Dimension of vectors
    window=3,             # Context window
    min_count=1,          # Minimum word frequency
    workers=4             # Parallel processing
)

# Access word vectors
print(model.wv['machine'])  # 100-D vector for "machine"

# Similarity between words
similarity = model.wv.similarity('learning', 'neural')
print(f"Similarity(learning, neural): {similarity:.3f}")

# Find most similar words
similar = model.wv.most_similar('data', topn=3)
print(f"Most similar to 'data':")
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Vector arithmetic (King - Man + Woman = Queen)
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(f"King - Man + Woman ≈ {result[0][0]}")  # Queen
```

### GloVe Vectors

```python
import numpy as np

# Load pretrained GloVe vectors
glove_vectors = {}
with open('glove.6B.100d.txt', 'r', encoding='utf8') as f:
    for line in f:
        word, vector = line.split(maxsplit=1)
        glove_vectors[word] = np.array(list(map(float, vector.split())))

# Use vectors
machine_vec = glove_vectors['machine']
print(f"'machine' vector shape: {machine_vec.shape}")

# Compute cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim = cosine_similarity(
    glove_vectors['learning'],
    glove_vectors['training']
)
print(f"Cosine similarity(learning, training): {sim:.3f}")
```

## 🤖 Advanced: Transformer Models

### BERT for Classification

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def classify(self, text):
        """Classify text using BERT"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][class_idx].item()
        
        return class_idx, confidence

# Usage
classifier = BERTClassifier()
text = "This product quality is excellent!"
prediction, confidence = classifier.classify(text)
print(f"Classification: {prediction}")  # 0 or 1
print(f"Confidence: {confidence:.1%}")
```

### Named Entity Recognition (NER)

```python
import spaCy

# Load English model
nlp = spacy.load('en_core_web_sm')

# Process text
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
doc = nlp(text)

# Extract entities
print("Entities:")
for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_}")

# Output:
# Entities:
#   Apple Inc.           → ORG (Organization)
#   Steve Jobs           → PERSON
#   Cupertino            → GPE (Geopolitical entity)
```

## 🚀 Production Pipeline

### Complete Text Processing Service

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained models
with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Comprehensive text analysis endpoint"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocessing
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess(text)
    
    # Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    # Classification
    text_vec = vectorizer.transform([text])
    prediction = sentiment_model.predict(text_vec)
    
    # NER
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return jsonify({
        'tokens': tokens,
        'sentiment': {
            'compound': sentiment['compound'],
            'label': 'POSITIVE' if sentiment['compound'] > 0 else 'NEGATIVE'
        },
        'classification': int(prediction[0]),
        'entities': entities
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## 💡 Interview Questions

### Q: TF-IDF vs Word Embeddings?
```
TF-IDF:
- Bag-of-words approach (word order ignored)
- Good for sparse, traditional ML
- Interpretable (feature importance visible)

Embeddings:
- Dense representations
- Capture semantic meaning
- Good for neural networks
- Context-aware (same word, different contexts)

Choice:
- TF-IDF + Logistic: Fast, simple baselines
- Embeddings + Deep: Complex patterns, better accuracy
```

### Q: How to handle imbalanced text data?
```
Solutions:
1. Class weighting: Give more importance to minority class
2. Oversampling: Duplicate minority examples
3. Undersampling: Reduce majority class
4. SMOTE: Generate synthetic examples
5. Focal loss: Penalize easy-to-classify examples more
```

## 🌟 Portfolio Value

✅ Classic ML + deep learning
✅ Multiple preprocessing techniques
✅ State-of-the-art transformers
✅ Production-grade architecture
✅ Diverse NLP tasks
✅ Real-world applications
✅ Solid fundamentals

## 📄 License

MIT License - Educational Use

---

**Development Ideas**:
1. Add chatbot with context persistence
2. Implement question-answering system
3. Machine translation (sequence-to-sequence)
4. Text summarization
5. Custom domain adaptation
