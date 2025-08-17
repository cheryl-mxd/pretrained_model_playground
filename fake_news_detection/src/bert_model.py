import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

class NewsDataset(Dataset):
    """Custom dataset for news classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTNewsClassifier:
    def __init__(self, model_name='bert-base-uncased', max_length=512, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(self, df, text_column='text', label_column='label', test_size=0.2):
        """Prepare data for BERT training"""
        print("Preparing data for BERT...")
        
        # Handle missing values more robustly
        print(f"Original dataset size: {len(df)}")
        
        # Check for missing values
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        
        if missing_text > 0:
            print(f"Warning: Found {missing_text} missing text values")
        if missing_labels > 0:
            print(f"Warning: Found {missing_labels} missing label values")
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=[text_column, label_column]).copy()
        print(f"Cleaned dataset size: {len(df_clean)}")
        
        # Additional cleaning for text data
        df_clean[text_column] = df_clean[text_column].astype(str)
        
        # Remove empty texts after string conversion
        non_empty_mask = df_clean[text_column].str.strip() != ""
        empty_count = (~non_empty_mask).sum()
        
        if empty_count > 0:
            print(f"Warning: Found {empty_count} empty texts. Removing them.")
            df_clean = df_clean[non_empty_mask].reset_index(drop=True)
        
        # Ensure we have enough data
        if len(df_clean) < 10:
            raise ValueError(f"Not enough valid data samples: {len(df_clean)}. Need at least 10.")
        
        # Get texts and labels
        texts = df_clean[text_column].tolist()
        labels = df_clean[label_column].tolist()
        
        # Validate that all texts are strings and not empty
        cleaned_texts = []
        cleaned_labels = []
        
        for text, label in zip(texts, labels):
            text_str = str(text).strip()
            if text_str and len(text_str) > 0:
                cleaned_texts.append(text_str)
                cleaned_labels.append(label)
        
        print(f"Final valid samples: {len(cleaned_texts)}")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            cleaned_texts, cleaned_labels, test_size=test_size, random_state=42, 
            stratify=cleaned_labels
        )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Create datasets
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset, train_texts, val_texts, train_labels, val_labels
    
    def create_data_loaders(self, train_dataset, val_dataset, batch_size=16):
        """Create data loaders"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def initialize_model(self, num_labels=2):
        """Initialize BERT model for classification"""
        print(f"Loading BERT model: {self.model_name}")
        
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        self.model.to(self.device)
        return self.model
    
    def predict_pretrained(self, texts, batch_size=32):
        """Use pre-trained BERT for sentiment analysis as baseline"""
        print("Using pre-trained BERT for prediction...")
        
        # Use sentiment analysis pipeline as proxy for news classification
        # Note: This is not ideal but demonstrates pre-trained model usage
        classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if self.device == 'cuda' else -1
        )
        
        predictions = []
        labels_map = {'NEGATIVE': 0, 'POSITIVE': 1}  # Map sentiment to fake/real
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i+batch_size]
            try:
                batch_preds = classifier(batch)
                for pred in batch_preds:
                    # Convert sentiment to fake/real news prediction
                    # This is a simplification - in practice you'd use a news-specific model
                    label = labels_map.get(pred['label'], 1)
                    predictions.append(label)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Fill with default predictions
                predictions.extend([1] * len(batch))
        
        return predictions
    
    def train_model(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Fine-tune BERT model"""
        print("="*60)
        print("FINE-TUNING BERT MODEL")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training history
        train_losses = []
        val_accuracies = []
        
        print(f"Training for {epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in train_pbar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Calculate average training loss
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_accuracy = self.evaluate_model(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Plot training history
        self.plot_training_history(train_losses, val_accuracies)
        
        return train_losses, val_accuracies
    
    def evaluate_model(self, data_loader):
        """Evaluate model on validation/test data"""
        self.model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    
    def detailed_evaluation(self, data_loader, true_labels, dataset_name="Test"):
        """Perform detailed evaluation with metrics"""
        print(f"\n{dataset_name} Set Evaluation:")
        print("-" * 30)
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        plt.title(f'BERT - {dataset_name} Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def plot_training_history(self, train_losses, val_accuracies):
        """Plot training history"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Validation accuracy
        ax2.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_baseline(self, bert_results, baseline_results):
        """Compare BERT results with baseline models"""
        print("\n" + "="*60)
        print("BERT vs BASELINE COMPARISON")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = {
            'BERT (Fine-tuned)': [
                bert_results['accuracy'],
                bert_results['precision'],
                bert_results['recall'],
                bert_results['f1']
            ]
        }
        
        # Add baseline results if provided
        if baseline_results is not None and hasattr(baseline_results, 'loc'):
            for model in baseline_results.columns:
                comparison_data[model] = [
                    baseline_results.loc['Accuracy', model],
                    baseline_results.loc['Precision', model],
                    baseline_results.loc['Recall', model],
                    baseline_results.loc['F1-Score', model]
                ]
        
        comparison_df = pd.DataFrame(
            comparison_data,
            index=['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )
        
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        # Plot comparison
        comparison_df.T.plot(kind='bar', figsize=(12, 6), rot=45)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def save_model(self, save_path="models/bert_finetuned"):
        """Save fine-tuned model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, model_path="models/bert_finetuned"):
        """Load fine-tuned model"""
        print(f"Loading model from: {model_path}")
        
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        print("Model loaded successfully")
    
    def predict_single_text(self, text):
        """Predict a single text sample"""
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        pred_label = prediction.item()
        confidence = probabilities[0][pred_label].item()
        
        label_map = {0: 'Fake News', 1: 'Real News'}
        
        return {
            'prediction': label_map[pred_label],
            'confidence': confidence,
            'probabilities': {
                'Fake News': probabilities[0][0].item(),
                'Real News': probabilities[0][1].item()
            }
        }

def main():
    """Main function to run BERT training and evaluation"""
    print("="*60)
    print("FAKE NEWS DETECTION - BERT MODEL")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv('data/processed_news.csv')
        print(f"Loaded processed data: {df.shape}")
    except FileNotFoundError:
        print("Processed data not found. Please run preprocess.py first.")
        return
    
    # Initialize BERT classifier
    bert_classifier = BERTNewsClassifier(
        model_name='bert-base-uncased',
        max_length=256  # Reduced for faster training
    )
    
    # For demonstration, use a smaller subset if dataset is large
    if len(df) > 1000:
        print(f"Using subset of {1000} samples for demonstration")
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # Prepare data
    train_dataset, val_dataset, train_texts, val_texts, train_labels, val_labels = bert_classifier.prepare_data(df)
    
    # Create data loaders
    train_loader, val_loader = bert_classifier.create_data_loaders(
        train_dataset, val_dataset, batch_size=8  # Small batch size for demo
    )
    
    # Option 1: Use pre-trained model for prediction (as baseline)
    print("\n" + "="*60)
    print("PRE-TRAINED MODEL EVALUATION")
    print("="*60)
    
    # Get predictions from pre-trained model (using sentiment as proxy)
    pretrained_predictions = bert_classifier.predict_pretrained(val_texts[:50])  # Small sample
    
    if len(pretrained_predictions) > 0:
        pretrained_accuracy = accuracy_score(val_labels[:len(pretrained_predictions)], pretrained_predictions)
        print(f"Pre-trained model accuracy: {pretrained_accuracy:.4f}")
    
    # Option 2: Fine-tune BERT
    print("\n" + "="*60)
    print("FINE-TUNING BERT")
    print("="*60)
    
    # Initialize model for fine-tuning
    bert_classifier.initialize_model(num_labels=2)
    
    # Train model (reduced epochs for demonstration)
    train_losses, val_accuracies = bert_classifier.train_model(
        train_loader, val_loader, 
        epochs=2,  # Reduced for demo
        learning_rate=2e-5
    )
    
    # Evaluate fine-tuned model
    bert_results = bert_classifier.detailed_evaluation(
        val_loader, val_labels, "Validation"
    )
    
    # Compare with baseline models
    try:
        baseline_results = pd.read_csv('models/baseline_results.csv', index_col=0)
        comparison_df = bert_classifier.compare_with_baseline(bert_results, baseline_results)
    except FileNotFoundError:
        print("Baseline results not found. Showing BERT results only.")
        comparison_df = bert_classifier.compare_with_baseline(bert_results, None)
    
    # Save fine-tuned model
    bert_classifier.save_model()
    
    # Demonstration of single text prediction
    print("\n" + "="*60)
    print("SINGLE TEXT PREDICTION DEMO")
    print("="*60)
    
    sample_texts = [
        "Breaking news: Scientists discover new treatment for cancer that shows promising results in clinical trials.",
        "SHOCKING: Celebrity spotted doing normal everyday activities! You won't believe what happened next!",
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nExample {i}: {text[:100]}...")
        result = bert_classifier.predict_single_text(text)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Fake={result['probabilities']['Fake News']:.4f}, "
              f"Real={result['probabilities']['Real News']:.4f}")
    
    print("\n" + "="*60)
    print("BERT MODEL TRAINING COMPLETED")
    print("="*60)
    
    return bert_classifier, bert_results, comparison_df

if __name__ == "__main__":
    main()