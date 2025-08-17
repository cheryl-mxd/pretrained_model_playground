import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


class NewsPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, true_path, fake_path):
        """Load and combine true and fake news datasets"""
        try:
            # Load datasets
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            
            # Add labels
            true_df['label'] = 1  # True news
            fake_df['label'] = 0  # Fake news
            
            # Combine datasets
            df = pd.concat([true_df, fake_df], ignore_index=True)
            
            print(f"Dataset loaded successfully!")
            print(f"True news articles: {len(true_df)}")
            print(f"Fake news articles: {len(fake_df)}")
            print(f"Total articles: {len(df)}")
            
            return df
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure the CSV files are in the data/ directory")
            return None
    
    def basic_eda(self, df):
        """Perform basic exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Label distribution
        print(f"\nLabel distribution:")
        label_counts = df['label'].value_counts()
        print(f"Real news (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
        print(f"Fake news (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
        
        # Plot label distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        df['label'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.title('Distribution of News Labels')
        plt.xlabel('Label (0=Fake, 1=Real)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Text length analysis
        df['text_length'] = df['text'].astype(str).apply(len)
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='text_length', by='label', ax=plt.gca())
        plt.title('Text Length Distribution by Label')
        plt.xlabel('Label (0=Fake, 1=Real)')
        plt.ylabel('Text Length (characters)')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('plots/text_length.png')
        # plt.show()
        
        # Statistical summary of text lengths
        print(f"\nText length statistics:")
        print(df.groupby('label')['text_length'].describe())
        
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Handle NaN values first
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # If text is empty after conversion, return empty string
        if not text.strip():
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Return empty string if cleaning resulted in empty text
        return text if text.strip() else ""
    
    def tokenize_text(self, text):
        """Tokenize text and remove stopwords"""
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def extract_linguistic_features(self, df):
        """Extract various linguistic features from the text"""
        print("\n" + "="*50)
        print("FEATURE EXTRACTION")
        print("="*50)
        
        features_df = df.copy()
        
        # Clean text
        print("Cleaning text...")
        features_df['cleaned_text'] = features_df['text'].apply(self.clean_text)
        
        # Basic text statistics
        print("Extracting basic text features...")
        features_df['word_count'] = features_df['cleaned_text'].apply(lambda x: len(x.split()))
        features_df['char_count'] = features_df['cleaned_text'].apply(len)
        features_df['sentence_count'] = features_df['text'].apply(lambda x: len(sent_tokenize(str(x))))
        features_df['avg_word_length'] = features_df['cleaned_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        features_df['avg_sentence_length'] = features_df['word_count'] / features_df['sentence_count']
        
        # POS tag features (sample for performance)
        print("Extracting POS tag features...")
        sample_size = min(1000, len(features_df))
        sample_indices = np.random.choice(len(features_df), sample_size, replace=False)
        
        pos_features = []
        for idx in sample_indices:
            text = features_df.iloc[idx]['cleaned_text']
            if text:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                pos_counts = Counter([tag for word, tag in pos_tags])
                pos_features.append({
                    'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0),
                    'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0),
                    'adj_count': pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),
                    'adv_count': pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
                })
            else:
                pos_features.append({'noun_count': 0, 'verb_count': 0, 'adj_count': 0, 'adv_count': 0})
        
        # Add POS features to sample
        pos_df = pd.DataFrame(pos_features, index=sample_indices)
        for col in pos_df.columns:
            features_df.loc[sample_indices, col] = pos_df[col]
        
        # Fill missing POS values with 0
        pos_cols = ['noun_count', 'verb_count', 'adj_count', 'adv_count']
        for col in pos_cols:
            if col not in features_df.columns:
                features_df[col] = 0
            features_df[col] = features_df[col].fillna(0)
        
        # Feature analysis
        print("\nFeature statistics by label:")
        feature_cols = ['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length']
        print(features_df.groupby('label')[feature_cols].mean())
        
        # Visualize features
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(feature_cols, 1):
            plt.subplot(2, 3, i)
            features_df.boxplot(column=feature, by='label', ax=plt.gca())
            plt.title(f'{feature.replace("_", " ").title()}')
            plt.xlabel('Label (0=Fake, 1=Real)')
            plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig('plots/features_by_label.png')
        # plt.show()
        
        return features_df
    
    def create_tfidf_features(self, df, max_features=5000):
        """Create TF-IDF features"""
        print(f"\nCreating TF-IDF features (max_features={max_features})...")
        
        # Ensure cleaned_text column exists and handle missing values
        if 'cleaned_text' not in df.columns:
            print("Creating cleaned_text column...")
            df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove any remaining NaN values and empty strings
        print("Handling missing and empty text values...")
        df['cleaned_text'] = df['cleaned_text'].fillna("")
        
        # Filter out empty texts after cleaning
        non_empty_mask = df['cleaned_text'].str.strip() != ""
        empty_count = (~non_empty_mask).sum()
        
        if empty_count > 0:
            print(f"Warning: Found {empty_count} empty texts after cleaning. These will be filled with placeholder text.")
            # Fill empty texts with a placeholder to avoid TF-IDF errors
            df.loc[~non_empty_mask, 'cleaned_text'] = "empty document placeholder"
        
        print(f"Processing {len(df)} documents for TF-IDF...")
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Fit and transform - this should now work without NaN errors
        try:
            tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
        except ValueError as e:
            print(f"Error during TF-IDF transformation: {e}")
            print("Attempting to fix by removing any remaining problematic entries...")
            
            # Double-check for any problematic entries
            cleaned_texts = df['cleaned_text'].tolist()
            for i, text in enumerate(cleaned_texts):
                if pd.isna(text) or text is None:
                    cleaned_texts[i] = "empty document placeholder"
                elif not isinstance(text, str):
                    cleaned_texts[i] = str(text)
                elif text.strip() == "":
                    cleaned_texts[i] = "empty document placeholder"
            
            # Try again with cleaned list
            tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Sample features: {feature_names[:10]}")
        
        return tfidf_matrix, tfidf, feature_names
    
    def analyze_top_words(self, df, tfidf_vectorizer, n_words=20):
        """Analyze top words for each class"""
        print(f"\nAnalyzing top {n_words} words for each class...")
        
        # Get TF-IDF scores for each class
        fake_indices = df[df['label'] == 0].index
        real_indices = df[df['label'] == 1].index
        
        # Transform texts
        tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_text'])
        
        # Get mean TF-IDF scores for each class
        fake_tfidf = np.mean(tfidf_matrix[fake_indices].toarray(), axis=0)
        real_tfidf = np.mean(tfidf_matrix[real_indices].toarray(), axis=0)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get top words for each class
        fake_top_indices = np.argsort(fake_tfidf)[::-1][:n_words]
        real_top_indices = np.argsort(real_tfidf)[::-1][:n_words]
        
        fake_top_words = [(feature_names[i], fake_tfidf[i]) for i in fake_top_indices]
        real_top_words = [(feature_names[i], real_tfidf[i]) for i in real_top_indices]
        
        print("\nTop words in FAKE news:")
        for word, score in fake_top_words[:10]:
            print(f"  {word}: {score:.4f}")
            
        print("\nTop words in REAL news:")
        for word, score in real_top_words[:10]:
            print(f"  {word}: {score:.4f}")
        
        # Visualize top words
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        words, scores = zip(*fake_top_words[:10])
        plt.barh(words, scores, color='red', alpha=0.7)
        plt.title('Top 10 Words in Fake News')
        plt.xlabel('Average TF-IDF Score')
        
        plt.subplot(1, 2, 2)
        words, scores = zip(*real_top_words[:10])
        plt.barh(words, scores, color='green', alpha=0.7)
        plt.title('Top 10 Words in Real News')
        plt.xlabel('Average TF-IDF Score')
        
        plt.tight_layout()
        plt.savefig('plots/top_words.png')
        # plt.show()
        
        return fake_top_words, real_top_words

def main():
    """Main function to run preprocessing and EDA"""
    print("="*60)
    print("FAKE NEWS DETECTION - DATA PREPROCESSING & EDA")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = NewsPreprocessor()
    
    # Load data (you'll need to adjust these paths)
    print("Please ensure your data files are in the data/ directory:")
    print("- data/True.csv")
    print("- data/Fake.csv")
    
    df = preprocessor.load_data('data/True.csv', 'data/Fake.csv')
    
    # Perform EDA
    df = preprocessor.basic_eda(df)
    
    # Extract features
    features_df = preprocessor.extract_linguistic_features(df)
    
    # Create TF-IDF features
    tfidf_matrix, tfidf_vectorizer, feature_names = preprocessor.create_tfidf_features(features_df)
    
    # Analyze top words
    fake_words, real_words = preprocessor.analyze_top_words(features_df, tfidf_vectorizer)
    
    # Save processed data
    features_df.to_csv('data/processed_news.csv', index=False)
    print("\nProcessed data saved to 'data/processed_news.csv'")
    
    return features_df, tfidf_matrix, tfidf_vectorizer

if __name__ == "__main__":
    main()