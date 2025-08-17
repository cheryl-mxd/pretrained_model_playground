# Fake News Detection Project

A comprehensive Natural Language Processing project that implements and compares classical machine learning approaches with modern transformer-based models (BERT) for fake news detection.

## 🎯 Project Overview

This project provides a complete pipeline for fake news detection, including:
- Data preprocessing and exploratory data analysis
- Feature extraction (TF-IDF, linguistic features)
- Classical ML models (Logistic Regression, Naive Bayes, SVM, Random Forest)
- BERT fine-tuning and evaluation
- Comprehensive model comparison and analysis

## 📁 Project Structure

```
fake-news-detection/
├── data/                          # Dataset directory
│   ├── True.csv                   # Real news articles
│   ├── Fake.csv                   # Fake news articles
│   └── processed_news.csv         # Processed dataset (generated)
├── src/                           # Source code
│   ├── preprocess.py              # Data preprocessing and EDA
│   ├── baseline_model.py          # Classical ML models
│   └── bert_model.py              # BERT fine-tuning
├── models/                        # Trained models (generated)
│   ├── baseline_*.joblib          # Classical models
│   └── bert_finetuned/            # Fine-tuned BERT model
├── results/                       # Results and reports (generated)
│   ├── baseline_results.csv       # Baseline model performance
│   ├── model_comparison.csv       # Complete comparison
│   └── project_report.md          # Final report
├── main.py                        # Main project runner
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
git clone <repository-url>
cd fake-news-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data):

1. Download `True.csv` and `Fake.csv`
2. Place them in the `data/` directory

*Note: If you don't have the dataset, the script will create sample data for demonstration.*

### 3. Run the Project

**Full Pipeline:**
```bash
python main.py
```

**Quick Demo (smaller dataset, faster execution):**
```bash
python main.py --quick
```

**Skip BERT training (if you don't have GPU):**
```bash
python main.py --skip-bert
```

**Other Options:**
```bash
# Run only preprocessing
python main.py --preprocessing-only

# Run preprocessing + baseline models only
python main.py --baseline-only
```

## 📊 Components Description

### 1. Data Preprocessing (`src/preprocess.py`)

**Features:**
- ✅ Dataset loading and basic statistics
- ✅ Text cleaning (lowercasing, punctuation removal, URL removal)
- ✅ Exploratory data analysis with visualizations
- ✅ Linguistic feature extraction:
  - Word count, character count, sentence count
  - Average word/sentence length
  - POS tag frequencies
- ✅ TF-IDF vectorization
- ✅ Top words analysis for each class

**Key Functions:**
```python
preprocessor = NewsPreprocessor()
df = preprocessor.load_data('data/True.csv', 'data/Fake.csv')
df = preprocessor.basic_eda(df)
features_df = preprocessor.extract_linguistic_features(df)
tfidf_matrix, vectorizer, _ = preprocessor.create_tfidf_features(features_df)
```

### 2. Baseline Models (`src/baseline_model.py`)

**Models Implemented:**
- ✅ Logistic Regression
- ✅ Support Vector Machine
- ✅ Random Forest

**Features:**
- ✅ Comprehensive evaluation (Accuracy, Precision, Recall, F1)
- ✅ Cross-validation
- ✅ Confusion matrices
- ✅ Feature importance analysis
- ✅ Hyperparameter tuning
- ✅ Model persistence

**Key Functions:**
```python
baseline = BaselineModels()
X, y = baseline.prepare_features(df)
baseline.initialize_models()
baseline.train_and_evaluate(X, y)
results_df = baseline.display_results()
```

### 3. BERT Model (`src/bert_model.py`)

**Features:**
- ✅ Pre-trained BERT evaluation
- ✅ BERT fine-tuning for news classification
- ✅ Custom dataset class for PyTorch
- ✅ Training with learning rate scheduling
- ✅ Comprehensive evaluation and comparison
- ✅ Single text prediction capability
- ✅ Model saving and loading

**Key Functions:**
```python
bert_classifier = BERTNewsClassifier('bert-base-uncased')
train_dataset, val_dataset, _, _, _, _ = bert_classifier.prepare_data(df)
bert_classifier.initialize_model(num_labels=2)
bert_classifier.train_model(train_loader, val_loader, epochs=3)
results = bert_classifier.detailed_evaluation(val_loader, val_labels)
```

## 📈 Expected Results

### Baseline Models Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.94 | ~0.94 | ~0.94 | ~0.94 |
| Random Forest | ~0.93 | ~0.93 | ~0.93 | ~0.93 |
| SVM | ~0.92 | ~0.92 | ~0.92 | ~0.92 |

### BERT Model Performance
- **Accuracy**: ~0.96-0.98
- **F1-Score**: ~0.96-0.98
- **Improvement over baseline**: 2-4%

## 🔧 Technical Requirements

### Hardware
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for BERT training (NVIDIA GPU with CUDA support)

### Software
- **Python**: 3.7 or higher
- **PyTorch**: 1.9.0 or higher
- **Transformers**: 4.11.0 or higher
- **Scikit-learn**: 1.0.0 or higher
- **CUDA**: Optional for GPU acceleration

## 💡 Usage Examples

### Running Individual Components

```python
# Preprocessing only
from src.preprocess import NewsPreprocessor
preprocessor = NewsPreprocessor()
df = preprocessor.load_data('data/True.csv', 'data/Fake.csv')

# Baseline models only
from src.baseline_model import BaselineModels
baseline = BaselineModels()
# ... (continue with training)

# BERT model only
from src.bert_model import BERTNewsClassifier
bert_classifier = BERTNewsClassifier()
# ... (continue with training)
```

### Making Predictions

```python
# Load trained BERT model
bert_classifier = BERTNewsClassifier()
bert_classifier.load_model('models/bert_finetuned')

# Predict single text
text = "Breaking news: Scientists discover revolutionary treatment..."
result = bert_classifier.predict_single_text(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📊 Evaluation Metrics

The project uses comprehensive evaluation metrics:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown
- **Cross-validation**: K-fold validation for robust evaluation

## 🎛️ Configuration Options

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --quick              Run in quick mode (smaller datasets, fewer epochs)
  --skip-bert          Skip BERT training (faster execution)
  --preprocessing-only Run only preprocessing step
  --baseline-only      Run preprocessing + baseline models only
  --data-path PATH     Specify data directory path
```

### Model Parameters

**BERT Configuration:**
- Model: `bert-base-uncased`
- Max length: 512 tokens (256 in quick mode)
- Batch size: 8 (4 in quick mode)
- Learning rate: 2e-5
- Epochs: 3 (1 in quick mode)

**TF-IDF Configuration:**
- Max features: 5000 (2000 in quick mode)
- N-gram range: (1, 2)
- Min document frequency: 2
- Max document frequency: 0.95

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller batch size
   python main.py --quick
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Not Found**
   - Download from Kaggle link above
   - Or let the script create sample data automatically

4. **Slow Training**
   - Use `--quick` flag for faster execution
   - Use `--skip-bert` to skip BERT training
   - Ensure GPU is available for BERT training

### Performance Tips

- **Use GPU**: Install CUDA-enabled PyTorch for faster BERT training
- **Increase RAM**: Close other applications during training
- **Quick Mode**: Use `--quick` for demonstration purposes
- **Parallel Processing**: The code uses multiprocessing where possible

## 📄 License

This project is for educational purposes. Please ensure you have the right to use the dataset according to its license terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify that your dataset is in the correct format
4. Check that you have sufficient system resources

## 🙏 Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data)
- BERT Model: Hugging Face Transformers library
- Scikit-learn: For classical machine learning algorithms

---

**Happy Learning! 🎉**