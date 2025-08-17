import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for classical ML models"""
        print("Preparing features for classical ML models...")
        
        # Text features using TF-IDF
        print("Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Ensure we have cleaned text and handle missing values
        if 'cleaned_text' not in df.columns:
            from preprocess import NewsPreprocessor
            preprocessor = NewsPreprocessor()
            print("Cleaning text data...")
            df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
        
        # Handle any remaining NaN or empty values
        df['cleaned_text'] = df['cleaned_text'].fillna("")
        empty_mask = df['cleaned_text'].str.strip() == ""
        if empty_mask.any():
            print(f"Warning: Found {empty_mask.sum()} empty texts. Filling with placeholder.")
            df.loc[empty_mask, 'cleaned_text'] = "empty document placeholder"
        
        # Create TF-IDF features
        try:
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
        except ValueError as e:
            print(f"Error in TF-IDF transformation: {e}")
            print("Cleaning text data more aggressively...")
            
            # More aggressive cleaning
            cleaned_texts = []
            for text in df['cleaned_text']:
                if pd.isna(text) or text is None or str(text).strip() == "":
                    cleaned_texts.append("empty document placeholder")
                else:
                    cleaned_texts.append(str(text))
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts).toarray()
        
        # Linguistic features
        print("Preparing linguistic features...")
        linguistic_features = []
        
        # Basic features that should exist
        feature_cols = ['word_count', 'char_count', 'sentence_count', 
                       'avg_word_length', 'avg_sentence_length']
        
        # Check which features exist and create missing ones
        for col in feature_cols:
            if col not in df.columns:
                if col == 'word_count':
                    df[col] = df['cleaned_text'].apply(lambda x: len(str(x).split()) if x and str(x).strip() else 0)
                elif col == 'char_count':
                    df[col] = df['cleaned_text'].apply(lambda x: len(str(x)) if x else 0)
                elif col == 'sentence_count':
                    df[col] = df['text'].apply(lambda x: max(1, len(str(x).split('.'))) if x else 1)
                elif col == 'avg_word_length':
                    df[col] = df['cleaned_text'].apply(
                        lambda x: np.mean([len(word) for word in str(x).split()]) if x and str(x).split() else 0
                    )
                elif col == 'avg_sentence_length':
                    df[col] = df['word_count'] / df['sentence_count']
        
        # Handle any missing or infinite values
        for col in feature_cols:
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        linguistic_features = df[feature_cols].values
        
        # Normalize linguistic features
        linguistic_features = self.scaler.fit_transform(linguistic_features)
        
        # Combine features
        X = np.hstack([tfidf_features, linguistic_features])
        y = df['label'].values
        
        print(f"Total feature dimensions: {X.shape[1]}")
        print(f"  - TF-IDF features: {tfidf_features.shape[1]}")
        print(f"  - Linguistic features: {linguistic_features.shape[1]}")
        
        return X, y
    
    def initialize_models(self):
        """Initialize classical ML models"""
        print("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ), 
            # 'SVM': SVC(
            #     random_state=self.random_state,
            #     probability=True
            # ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING AND EVALUATION")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def display_results(self):
        """Display comprehensive results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            name: [
                results['accuracy'],
                results['precision'], 
                results['recall'],
                results['f1'],
                results['cv_mean']
            ] for name, results in self.results.items()
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV F1-Score'])
        
        print("\nModel Performance Summary:")
        print(results_df.round(4))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1'])
        print(f"\nBest performing model: {best_model_name}")
        print(f"F1-Score: {self.results[best_model_name]['f1']:.4f}")
        
        # Plot results
        self.plot_results(results_df)
        
        # Show confusion matrices
        self.plot_confusion_matrices()
        
        # Show detailed classification reports
        self.show_classification_reports()
        
        return results_df
    
    def plot_results(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(results_df.columns, results_df.loc[metric], 
                         color=colors[i], alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Fake', 'Real'],
                       yticklabels=['Fake', 'Real'])
            axes[i].set_title(f'{name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def show_classification_reports(self):
        """Show detailed classification reports"""
        for name, results in self.results.items():
            print(f"\n{name} - Detailed Classification Report:")
            print("-" * 50)
            target_names = ['Fake News', 'Real News']
            print(classification_report(results['y_test'], results['y_pred'], 
                                      target_names=target_names))
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Logistic Regression'):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {model_name}")
        print("="*60)
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        print(f"Tuning {model_name} with parameters: {param_grid}")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        for name, results in self.results.items():
            model = results['model']
            
            if hasattr(model, 'feature_importances_'):
                print(f"\n{name} - Top 20 Most Important Features:")
                print("-" * 50)
                
                # Get feature names (TF-IDF features + linguistic features)
                tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
                linguistic_features = ['word_count', 'char_count', 'sentence_count', 
                                     'avg_word_length', 'avg_sentence_length']
                all_features = list(tfidf_features) + linguistic_features
                
                # Get importance scores
                importances = model.feature_importances_
                
                # Sort by importance
                feature_importance = list(zip(all_features, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Display top features
                for i, (feature, importance) in enumerate(feature_importance[:20]):
                    print(f"{i+1:2d}. {feature:30s}: {importance:.6f}")
                
                # Plot feature importance
                top_features = feature_importance[:15]
                features, scores = zip(*top_features)
                
                plt.figure(figsize=(10, 8))
                plt.barh(features, scores)
                plt.xlabel('Feature Importance')
                plt.title(f'{name} - Top 15 Feature Importances')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
            
            elif hasattr(model, 'coef_'):
                print(f"\n{name} - Top 20 Most Important Features (by coefficient magnitude):")
                print("-" * 50)
                
                # Get feature names
                tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
                linguistic_features = ['word_count', 'char_count', 'sentence_count', 
                                     'avg_word_length', 'avg_sentence_length']
                all_features = list(tfidf_features) + linguistic_features
                
                # Get coefficients (for binary classification, take first row)
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                
                # Sort by absolute coefficient value
                feature_coef = list(zip(all_features, coef, np.abs(coef)))
                feature_coef.sort(key=lambda x: x[2], reverse=True)
                
                # Display top features
                for i, (feature, coef_val, abs_coef) in enumerate(feature_coef[:20]):
                    direction = "→ Real" if coef_val > 0 else "→ Fake"
                    print(f"{i+1:2d}. {feature:30s}: {coef_val:8.4f} {direction}")
    
    def save_models(self, filepath_prefix="models/baseline"):
        """Save trained models"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        print(f"\nSaving models to {filepath_prefix}_*")
        
        # Save models
        for name, results in self.results.items():
            model_filename = f"{filepath_prefix}_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(results['model'], model_filename)
            print(f"  Saved {name}: {model_filename}")
        
        # Save vectorizer and scaler
        joblib.dump(self.tfidf_vectorizer, f"{filepath_prefix}_tfidf_vectorizer.joblib")
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.joblib")
        print(f"  Saved TF-IDF vectorizer and scaler")
        
        # Save results summary
        results_summary = {
            name: {k: v for k, v in results.items() 
                  if k not in ['model', 'y_test', 'y_pred', 'y_pred_proba']}
            for name, results in self.results.items()
        }
        
        import json
        with open(f"{filepath_prefix}_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"  Saved results summary: {filepath_prefix}_results.json")

def main():
    """Main function to run baseline model training"""
    print("="*60)
    print("FAKE NEWS DETECTION - BASELINE MODELS")
    print("="*60)
    
    # Load processed data
    try:
        df = pd.read_csv('data/processed_news.csv')
        print(f"Loaded processed data: {df.shape}")
    except FileNotFoundError:
        print("Processed data not found. Running preprocessing first...")
        from preprocess import main as preprocess_main
        df, _, _ = preprocess_main()
    
    # Initialize baseline models
    baseline = BaselineModels(random_state=42)
    
    # Prepare features
    X, y = baseline.prepare_features(df)
    
    # Initialize models
    baseline.initialize_models()
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = baseline.train_and_evaluate(X, y)
    
    # Display results
    results_df = baseline.display_results()
    
    # Feature importance analysis
    baseline.feature_importance_analysis()
    
    # Hyperparameter tuning for best models
    print("\nPerforming hyperparameter tuning for top models...")
    for model_name in ['Logistic Regression', 'Random Forest']:
        if model_name in baseline.models:
            baseline.hyperparameter_tuning(X_train, y_train, model_name)
    
    # Re-evaluate after tuning
    print("\nRe-evaluating after hyperparameter tuning...")
    baseline.train_and_evaluate(X, y)
    baseline.display_results()
    
    # Save models
    baseline.save_models()
    
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("="*60)
    
    return baseline, results_df

if __name__ == "__main__":
    main()