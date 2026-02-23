# 📚 NLP Projects - Natural Language Processing

A **comprehensive guide to NLP techniques** including text preprocessing, sentiment analysis, topic modeling, named entity recognition, and transformer models.

## 🎯 Overview

This project covers:
- ✅ Text preprocessing & tokenization
- ✅ Word embeddings (Word2Vec, GloVe)
- ✅ Sentiment analysis
- ✅ Named Entity Recognition (NER)
- ✅ Topic modeling (LDA)
- ✅ Text classification
- ✅ Transformer models (BERT, GPT)

## 🔤 Text Preprocessing

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

class TextPreprocessor:
    """Clean and prepare text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text
    
    def tokenize(self, text):
        """Split into tokens"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove common words"""
        filtered = [token for token in tokens if token not in self.stop_words]
        return filtered
    
    def stem(self, tokens):
        """Reduce to stem"""
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return stemmed
    
    def lemmatize(self, tokens):
        """Reduce to lemma"""
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def preprocess_pipeline(self, text):
        """Full preprocessing"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens
```

## 💭 Sentiment Analysis

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from transformers import pipeline

class SentimentAnalyzer:
    """Analyze sentiment"""
    
    @staticmethod
    def textblob_sentiment(text):
        """Simple sentiment using TextBlob"""
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity  # -1 to 1
        subjectivity = analysis.sentiment.subjectivity  # 0 to 1
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    @staticmethod
    def transformer_sentiment(text):
        """Use BERT for sentiment"""
        classifier = pipeline('sentiment-analysis', 
                            model='distilbert-base-uncased-finetuned-sst-2-english')
        
        result = classifier(text)
        return result[0]  # {'label': 'POSITIVE/NEGATIVE', 'score': probability}
    
    @staticmethod
    def train_classifier(texts, labels):
        """Train custom classifier"""
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        
        clf = MultinomialNB()
        clf.fit(X, labels)
        
        return {
            'vectorizer': vectorizer,
            'classifier': clf
        }
```

## 🏷️ Named Entity Recognition

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
import spacy

class EntityRecognizer:
    """Extract named entities"""
    
    @staticmethod
    def spacy_ner(text):
        """Using spaCy"""
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    @staticmethod
    def bert_ner(text):
        """Using BERT for NER"""
        nlp = hf_pipeline('ner', model='dslim/bert-base-uncased-finetuned-ner')
        
        results = nlp(text)
        return results
    
    @staticmethod
    def extract_entities(doc, entity_types=['PERSON', 'ORG', 'GPE']):
        """Filter entity types"""
        filtered = [
            {
                'text': ent.text,
                'label': ent.label_
            }
            for ent in doc.ents if ent.label_ in entity_types
        ]
        return filtered
```

## 🎯 Topic Modeling

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    """Extract topics from documents"""
    
    @staticmethod
    def lda_topics(documents, n_topics=5, n_words=10):
        """Latent Dirichlet Allocation"""
        # Vectorize
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic': topic_idx,
                'words': top_words,
                'weights': topic[top_indices]
            })
        
        return topics, lda, vectorizer
```

## 📊 Text Classification

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

class TextClassifier:
    """Classify text"""
    
    @staticmethod
    def create_pipeline(classifier_type='nb'):
        """Create classification pipeline"""
        if classifier_type == 'nb':
            clf = MultinomialNB()
        elif classifier_type == 'svm':
            clf = LinearSVC(random_state=42, max_iter=1000)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', clf)
        ])
        
        return pipeline
    
    @staticmethod
    def train_and_evaluate(X_train, y_train, X_test, y_test):
        """Train and test"""
        pipeline = TextClassifier.create_pipeline('svm')
        pipeline.fit(X_train, y_train)
        
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'model': pipeline
        }
```

## 🤖 Transformer Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TransformerNLP:
    """Use transformer models"""
    
    @staticmethod
    def text_generation():
        """Generate text"""
        from transformers import pipeline
        
        generator = pipeline('text-generation', model='gpt2')
        texts = generator("Machine learning is", max_length=50, num_return_sequences=3)
        
        return texts
    
    @staticmethod
    def question_answering(context, question):
        """Extract answers from context"""
        from transformers import pipeline
        
        qa = pipeline('question-answering', model='bert-base-uncased')
        
        answer = qa(question=question, context=context)
        return answer
    
    @staticmethod
    def summarization(text):
        """Summarize text"""
        from transformers import pipeline
        
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        summary = summarizer(text, max_length=100, min_length=50)
        
        return summary
```

## 💡 Interview Talking Points

**Q: Difference between stemming and lemmatization?**
```
Answer:
- Stemming: Crude removal of suffixes (run, running → runn)
- Lemmatization: Dictionary-based reduction (better, good → good)
- Stemming faster but less accurate
- Lemmatization more accurate but slower
```

**Q: Word embeddings vs TF-IDF?**
```
Answer:
- TF-IDF: Frequency-based, sparse, interpretable
- Word2Vec: Dense vectors, semantic meaning
- TF-IDF: Good for traditional ML
- Embeddings: Required for deep learning
```

## 🌟 Portfolio Value

✅ Text preprocessing
✅ Sentiment analysis
✅ NER (Named Entity Recognition)
✅ Topic modeling
✅ Text classification
✅ Transformer models
✅ NLP applications

---

**Technologies**: NLTK, spaCy, Scikit-learn, Transformers, PyTorch

