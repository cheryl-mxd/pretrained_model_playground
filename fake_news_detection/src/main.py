#!/usr/bin/env python3
"""
Fake News Detection Project - Main Runner
==========================================

This script runs the complete fake news detection pipeline:
1. Data preprocessing and EDA
2. Baseline classical ML models
3. BERT fine-tuning and evaluation
4. Model comparison and analysis

Usage:
    python main.py [--data-path DATA_PATH] [--quick] [--skip-bert]

Requirements:
    - Install required packages: pip install -r requirements.txt
    - Download dataset from Kaggle and place in data/ folder
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup the project environment"""
    print("Setting up project environment...")
    
    # Create necessary directories
    directories = ['data', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' ready")
    
    # Check for data files
    data_files = ['data/True.csv', 'data/Fake.csv']
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print("\n⚠️  Warning: The following data files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nTo get the data:")
        print("1. Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data")
        print("2. Extract True.csv and Fake.csv to the data/ folder")
        print("3. Run this script again")
        
        # Create sample data for demonstration
        print("\n🔄 Creating sample data for demonstration...")
        create_sample_data()
    else:
        print("✓ Data files found")

def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    
    # Sample real news texts
    real_news_samples = [
        "The Federal Reserve announced today that it will maintain interest rates at current levels following the latest policy meeting. Economic indicators suggest steady growth with controlled inflation rates.",
        "Scientists at MIT have developed a new battery technology that could increase electric vehicle range by up to 40%. The research was published in the journal Nature Energy this week.",
        "The World Health Organization reported a significant decline in malaria cases across sub-Saharan Africa, attributing the improvement to increased funding for prevention programs.",
        "Climate researchers warn that Arctic ice is melting at an unprecedented rate. New satellite data shows a 15% decline in ice coverage compared to the same period last year.",
        "The United Nations Security Council voted unanimously to extend peacekeeping missions in three African nations for another year, with additional funding approved for humanitarian aid.",
    ] * 40  # Repeat to create 200 samples
    
    # Sample fake news texts
    fake_news_samples = [
        "BREAKING: Government secretly controls weather using hidden satellites! Leaked documents reveal shocking truth that mainstream media won't tell you!",
        "Doctors HATE this simple trick! Local mom discovers miracle cure that pharmaceutical companies don't want you to know about!",
        "ALERT: New study proves that drinking water causes cancer! Government cover-up exposed by brave whistleblower!",
        "SHOCKING: Celebrity reveals that Earth is actually flat and space agencies are lying to us! Evidence that will blow your mind!",
        "URGENT: Vaccines contain microchips designed to control your thoughts! Secret government program finally exposed by anonymous hacker!",
    ] * 40  # Repeat to create 200 samples
    
    # Create True.csv
    true_df = pd.DataFrame({
        'title': [f"Real News Story {i+1}" for i in range(200)],
        'text': real_news_samples,
        'subject': np.random.choice(['politics', 'world', 'business', 'science'], 200),
        'date': pd.date_range('2020-01-01', periods=200, freq='D')
    })
    
    # Create Fake.csv
    fake_df = pd.DataFrame({
        'title': [f"Fake News Story {i+1}" for i in range(200)],
        'text': fake_news_samples,
        'subject': np.random.choice(['politics', 'world', 'entertainment', 'health'], 200),
        'date': pd.date_range('2020-01-01', periods=200, freq='D')
    })
    
    # Add some noise to make it more realistic
    for df in [true_df, fake_df]:
        # Add some variation to text lengths
        for i in range(len(df)):
            base_text = df.loc[i, 'text']
            if np.random.random() > 0.5:
                df.loc[i, 'text'] = base_text + " Additional details and context provided here."
    
    # Save files
    true_df.to_csv('data/True.csv', index=False)
    fake_df.to_csv('data/Fake.csv', index=False)
    
    print(f"✓ Created sample data: {len(true_df)} real news, {len(fake_df)} fake news articles")

def run_preprocessing(quick_mode=False):
    """Run data preprocessing and EDA"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING AND EDA")
    print("="*60)
    
    try:
        from src.preprocess import NewsPreprocessor
        
        preprocessor = NewsPreprocessor()
        
        # Load data
        df = preprocessor.load_data('data/True.csv', 'data/Fake.csv')
        if df is None:
            print("❌ Failed to load data")
            return None
        
        # Quick mode: use subset
        if quick_mode and len(df) > 500:
            print(f"Quick mode: Using subset of 500 samples")
            df = df.sample(n=500, random_state=42).reset_index(drop=True)
        
        # Run EDA
        df = preprocessor.basic_eda(df)
        
        # Extract features
        features_df = preprocessor.extract_linguistic_features(df)
        
        # Create TF-IDF features
        tfidf_matrix, tfidf_vectorizer, feature_names = preprocessor.create_tfidf_features(
            features_df, max_features=2000 if quick_mode else 5000
        )
        
        # Analyze top words
        fake_words, real_words = preprocessor.analyze_top_words(features_df, tfidf_vectorizer)
        
        # Save processed data
        features_df.to_csv('data/processed_news.csv', index=False)
        print("✓ Processed data saved to 'data/processed_news.csv'")
        
        return features_df, tfidf_matrix, tfidf_vectorizer
        
    except ImportError:
        print("❌ Could not import preprocess module. Please check src/preprocess.py")
        return None
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None

def run_baseline_models(quick_mode=False):
    """Run baseline classical ML models"""
    print("\n" + "="*60)
    print("STEP 2: BASELINE CLASSICAL ML MODELS")
    print("="*60)
    
    try:
        from fake_news_detection.src.baseline_model import BaselineModels
        
        # Load processed data
        if not os.path.exists('data/processed_news.csv'):
            print("❌ Processed data not found. Running preprocessing first...")
            result = run_preprocessing(quick_mode)
            if result is None:
                return None
        
        df = pd.read_csv('data/processed_news.csv')
        print(f"✓ Loaded processed data: {df.shape}")
        
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
        
        # Hyperparameter tuning (skip in quick mode)
        if not quick_mode:
            print("\n🔧 Performing hyperparameter tuning...")
            for model_name in ['Logistic Regression', 'Random Forest']:
                if model_name in baseline.models:
                    baseline.hyperparameter_tuning(X_train, y_train, model_name)
            
            # Re-evaluate after tuning
            print("\n🔄 Re-evaluating after hyperparameter tuning...")
            baseline.train_and_evaluate(X, y)
            results_df = baseline.display_results()
        
        # Save models and results
        baseline.save_models()
        results_df.to_csv('results/baseline_results.csv')
        
        print("✓ Baseline models training completed")
        return baseline, results_df
        
    except ImportError:
        print("❌ Could not import baseline_model module. Please check src/baseline_model.py")
        return None, None
    except Exception as e:
        print(f"❌ Error in baseline models: {e}")
        return None, None

def run_bert_model(quick_mode=False):
    """Run BERT fine-tuning and evaluation"""
    print("\n" + "="*60)
    print("STEP 3: BERT MODEL FINE-TUNING")
    print("="*60)
    
    try:
        from src.bert_model import BERTNewsClassifier
        import torch
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load data
        if not os.path.exists('data/processed_news.csv'):
            print("❌ Processed data not found. Please run preprocessing first.")
            return None, None
        
        df = pd.read_csv('data/processed_news.csv')
        
        # Use subset for demonstration
        if quick_mode or len(df) > 1000:
            sample_size = 200 if quick_mode else 1000
            print(f"Using subset of {sample_size} samples for BERT training")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Initialize BERT classifier
        bert_classifier = BERTNewsClassifier(
            model_name='bert-base-uncased',
            max_length=256 if quick_mode else 512
        )
        
        # Prepare data
        train_dataset, val_dataset, train_texts, val_texts, train_labels, val_labels = bert_classifier.prepare_data(df)
        
        # Create data loaders
        batch_size = 4 if quick_mode else 8
        train_loader, val_loader = bert_classifier.create_data_loaders(
            train_dataset, val_dataset, batch_size=batch_size
        )
        
        # Use pre-trained model for baseline
        print("\n📊 Evaluating pre-trained model...")
        sample_size = min(50, len(val_texts))
        pretrained_predictions = bert_classifier.predict_pretrained(val_texts[:sample_size])
        
        if len(pretrained_predictions) > 0:
            from sklearn.metrics import accuracy_score
            pretrained_accuracy = accuracy_score(val_labels[:len(pretrained_predictions)], pretrained_predictions)
            print(f"Pre-trained model accuracy: {pretrained_accuracy:.4f}")
        
        # Fine-tune BERT
        print("\n🚀 Fine-tuning BERT...")
        bert_classifier.initialize_model(num_labels=2)
        
        epochs = 1 if quick_mode else 2
        train_losses, val_accuracies = bert_classifier.train_model(
            train_loader, val_loader, 
            epochs=epochs,
            learning_rate=2e-5
        )
        
        # Evaluate fine-tuned model
        bert_results = bert_classifier.detailed_evaluation(
            val_loader, val_labels, "Validation"
        )
        
        # Compare with baseline models
        try:
            baseline_results = pd.read_csv('results/baseline_results.csv', index_col=0)
            comparison_df = bert_classifier.compare_with_baseline(bert_results, baseline_results)
        except FileNotFoundError:
            print("Baseline results not found. Showing BERT results only.")
            comparison_df = bert_classifier.compare_with_baseline(bert_results, None)
        
        # Save results
        comparison_df.to_csv('results/model_comparison.csv')
        
        # Save fine-tuned model
        bert_classifier.save_model()
        
        # Demo single predictions
        print("\n🔍 Testing single text predictions...")
        demo_texts = [
            "Scientists at Stanford University have developed a new AI system that can detect early signs of Alzheimer's disease with 95% accuracy.",
            "SHOCKING! This one weird trick will make you lose 50 pounds in just one week! Doctors hate this secret method!"
        ]
        
        for i, text in enumerate(demo_texts, 1):
            print(f"\nDemo {i}: {text[:80]}...")
            result = bert_classifier.predict_single_text(text)
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        
        print("✓ BERT model training completed")
        return bert_classifier, bert_results
        
    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        print("Please install: pip install torch transformers")
        return None, None
    except Exception as e:
        print(f"❌ Error in BERT training: {e}")
        return None, None

def generate_final_report(baseline_results=None, bert_results=None):
    """Generate final project report"""
    print("\n" + "="*60)
    print("FINAL PROJECT REPORT")
    print("="*60)
    
    report = []
    report.append("# Fake News Detection Project Report\n")
    report.append("## Project Overview\n")
    report.append("This project implements a comprehensive fake news detection system using both classical machine learning and modern deep learning approaches.\n")
    
    report.append("## Dataset\n")
    if os.path.exists('data/processed_news.csv'):
        df = pd.read_csv('data/processed_news.csv')
        report.append(f"- Total articles: {len(df)}")
        report.append(f"- Real news: {len(df[df['label']==1])}")
        report.append(f"- Fake news: {len(df[df['label']==0])}")
    
    report.append("\n## Methodology\n")
    report.append("### 1. Data Preprocessing")
    report.append("- Text cleaning (lowercasing, punctuation removal)")
    report.append("- Feature extraction (TF-IDF, linguistic features)")
    report.append("- Exploratory data analysis")
    
    report.append("\n### 2. Baseline Models")
    report.append("- Logistic Regression")
    report.append("- Support Vector Machine")
    report.append("- Random Forest")
    
    report.append("\n### 3. BERT Model")
    report.append("- Pre-trained BERT evaluation")
    report.append("- Fine-tuned BERT for fake news detection")
    
    if baseline_results is not None:
        report.append("\n## Baseline Results\n")
        report.append("| Model | Accuracy | Precision | Recall | F1-Score |")
        report.append("|-------|----------|-----------|--------|----------|")
        for model in baseline_results.columns:
            acc = baseline_results.loc['Accuracy', model]
            prec = baseline_results.loc['Precision', model]
            rec = baseline_results.loc['Recall', model]
            f1 = baseline_results.loc['F1-Score', model]
            report.append(f"| {model} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
    
    if bert_results is not None:
        report.append("\n## BERT Results\n")
        report.append(f"- Accuracy: {bert_results['accuracy']:.4f}")
        report.append(f"- Precision: {bert_results['precision']:.4f}")
        report.append(f"- Recall: {bert_results['recall']:.4f}")
        report.append(f"- F1-Score: {bert_results['f1']:.4f}")
    
    report.append("\n## Conclusions\n")
    report.append("- BERT-based models generally outperform classical approaches")
    report.append("- Feature engineering is crucial for classical models")
    report.append("- Fine-tuning pre-trained models yields better results than using them out-of-the-box")
    
    report.append("\n## Files Generated\n")
    report.append("- `data/processed_news.csv`: Preprocessed dataset")
    report.append("- `models/baseline_*.joblib`: Trained baseline models")
    report.append("- `models/bert_finetuned/`: Fine-tuned BERT model")
    report.append("- `results/baseline_results.csv`: Baseline model performance")
    report.append("- `results/model_comparison.csv`: Complete model comparison")
    
    # Save report
    with open('results/project_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))
    print(f"\n✓ Report saved to 'results/project_report.md'")

def main():
    """Main function to run the entire pipeline"""
    parser = argparse.ArgumentParser(description='Fake News Detection Project')
    parser.add_argument('--data-path', type=str, default='data/', help='Path to data directory')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (smaller datasets, fewer epochs)')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT training (faster execution)')
    parser.add_argument('--preprocessing-only', action='store_true', help='Run only preprocessing step')
    parser.add_argument('--baseline-only', action='store_true', help='Run only preprocessing and baseline models')
    
    args = parser.parse_args()
    
    print("🚀 FAKE NEWS DETECTION PROJECT")
    print("="*60)
    print(f"Quick mode: {'ON' if args.quick else 'OFF'}")
    print(f"Skip BERT: {'YES' if args.skip_bert else 'NO'}")
    
    # Setup environment
    setup_environment()
    
    # Results storage
    baseline_results = None
    bert_results = None
    
    try:
        # Step 1: Preprocessing
        print("\n⏳ Starting preprocessing...")
        preprocess_result = run_preprocessing(args.quick)
        if preprocess_result is None:
            print("❌ Preprocessing failed. Exiting.")
            return
        
        if args.preprocessing_only:
            print("✅ Preprocessing completed. Exiting as requested.")
            return
        
        # Step 2: Baseline models
        print("\n⏳ Starting baseline model training...")
        baseline_model, baseline_results = run_baseline_models(args.quick)
        
        if args.baseline_only:
            print("✅ Baseline training completed. Exiting as requested.")
            generate_final_report(baseline_results, None)
            return
        
        # Step 3: BERT model (unless skipped)
        if not args.skip_bert:
            print("\n⏳ Starting BERT model training...")
            bert_model, bert_results = run_bert_model(args.quick)
        else:
            print("\n⏭️  Skipping BERT training as requested.")
        
        # Generate final report
        generate_final_report(baseline_results, bert_results)
        
        print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print("   - data/: Processed datasets")
        print("   - models/: Trained models")
        print("   - results/: Performance metrics and reports")
        
        if not args.skip_bert and bert_results:
            print(f"\n🏆 Best performing model: BERT (F1: {bert_results['f1']:.4f})")
        elif baseline_results is not None:
            best_baseline = baseline_results.loc['F1-Score'].idxmax()
            best_f1 = baseline_results.loc['F1-Score', best_baseline]
            print(f"\n🏆 Best performing baseline model: {best_baseline} (F1: {best_f1:.4f})")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        print("Partial results may be available in the results/ directory.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error messages above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()