# Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        A[Twitter Training CSV]
        B[Twitter Validation CSV]
    end
    
    subgraph "Data Processing Layer"
        C[DataLoader]
        D[TextPreprocessor]
        E[FeatureEngineer]
    end
    
    subgraph "Machine Learning Layer"
        F[ModelTrainer]
        G[SyntheticDataGenerator]
        H[ModelEvaluator]
    end
    
    subgraph "Output Layer"
        I[Trained Models]
        J[Performance Reports]
        K[Visualizations]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> F
    F --> I
    H --> J
    H --> K
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Raw Data"
        A1[twitter_training.csv]
        A2[twitter_validation.csv]
    end
    
    subgraph "Unified Data"
        B1[raw_tweets.csv]
    end
    
    subgraph "Processed Data"
        C1[clean_tweets.csv]
        C2[Feature Extraction]
    end
    
    subgraph "Feature Matrices"
        D1[BoW Features]
        D2[TF-IDF Features]
    end
    
    subgraph "Model Training"
        E1[Multinomial NB]
        E2[Random Forest]
        E3[XGBoost]
    end
    
    subgraph "Synthetic Data"
        F1[SMOTE Augmentation]
        F2[TVAE Generation]
    end
    
    subgraph "Final Models"
        G1[Best Model Selection]
        G2[Model Persistence]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> C1
    C1 --> C2
    C2 --> D1
    C2 --> D2
    D1 --> E1
    D1 --> E2
    D1 --> E3
    D2 --> E1
    D2 --> E2
    D2 --> E3
    D1 --> F1
    D2 --> F1
    C1 --> F2
    F1 --> G1
    E1 --> G1
    E2 --> G1
    E3 --> G1
    G1 --> G2
```

## Component Interaction Diagram

```mermaid
classDiagram
    class DataLoader {
        +load_and_unify_data()
        +save_raw_data()
        -_create_unified_schema()
    }
    
    class TextPreprocessor {
        +clean_dataset()
        +save_clean_data()
        -_clean_text()
        -_extract_features()
        -_lemmatize_text()
    }
    
    class FeatureEngineer {
        +create_train_test_split()
        +create_bow_features()
        +create_tfidf_features()
        +save_features()
        +load_features()
    }
    
    class ModelTrainer {
        +train_all_models()
        +get_best_model()
        +retrain_on_augmented_data()
        +save_best_model()
        +save_model_card()
    }
    
    class SyntheticDataGenerator {
        +analyze_class_imbalance()
        +apply_smote()
        +generate_tvae_data()
        +save_synthetic_data()
    }
    
    class ModelEvaluator {
        +evaluate_model()
        +plot_confusion_matrix()
        +generate_classification_report()
        +save_evaluation_report()
    }
    
    DataLoader --> TextPreprocessor
    TextPreprocessor --> FeatureEngineer
    FeatureEngineer --> ModelTrainer
    FeatureEngineer --> SyntheticDataGenerator
    ModelTrainer --> ModelEvaluator
    SyntheticDataGenerator --> ModelTrainer
```

## Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant Main as run_pipeline.py
    participant DL as DataLoader
    participant TP as TextPreprocessor
    participant FE as FeatureEngineer
    participant MT as ModelTrainer
    participant SG as SyntheticDataGenerator
    participant ME as ModelEvaluator
    
    Main->>DL: load_and_unify_data()
    DL-->>Main: unified_dataset
    
    Main->>TP: clean_dataset()
    TP-->>Main: clean_dataset
    
    Main->>FE: create_train_test_split()
    FE-->>Main: X_train, X_test, y_train, y_test
    
    Main->>FE: create_bow_features()
    FE-->>Main: bow_features
    
    Main->>FE: create_tfidf_features()
    FE-->>Main: tfidf_features
    
    Main->>MT: train_all_models()
    MT-->>Main: training_results
    
    Main->>MT: get_best_model()
    MT-->>Main: best_model_key
    
    Main->>SG: apply_smote()
    SG-->>Main: X_augmented, y_augmented
    
    Main->>MT: retrain_on_augmented_data()
    MT-->>Main: augmented_results
    
    Main->>ME: evaluate_model()
    ME-->>Main: evaluation_metrics
    
    Main->>ME: plot_confusion_matrix()
    ME-->>Main: visualization_saved
```

## File System Architecture

```
SocialMedia_SentimentAnalysis/
├── datasets/                    # Data Storage Layer
│   ├── twitter_training.csv    # Input: Training data
│   ├── twitter_validation.csv  # Input: Validation data
│   ├── raw_tweets.csv          # Processed: Unified raw data
│   ├── clean_tweets.csv        # Processed: Clean data
│   └── synthetic_tvae_data.csv # Generated: Synthetic data
│
├── features/                    # Feature Storage Layer
│   ├── bow.npz                 # BoW feature matrices
│   ├── tfidf.npz              # TF-IDF feature matrices
│   ├── bow_vocab.pkl          # BoW vocabulary
│   └── tfidf_vocab.pkl        # TF-IDF vocabulary
│
├── models/                      # Model Storage Layer
│   ├── best_model_*.pkl        # Serialized models
│   └── model_card.json         # Model metadata
│
├── reports/                     # Output Layer
│   ├── eda/                    # Exploratory Data Analysis
│   │   ├── overview_analysis.png
│   │   └── sentiment_wordclouds.png
│   └── evaluation/             # Model Evaluation
│       └── confusion_matrices.png
│
├── src/                         # Core Logic Layer
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion
│   ├── preprocessor.py         # Text processing
│   ├── feature_engineer.py     # Feature creation
│   ├── model_trainer.py        # ML training
│   ├── synthetic_generator.py  # Data augmentation
│   └── evaluator.py           # Performance evaluation
│
├── tests/                       # Testing Layer
│   ├── test_pipeline.py        # Integration tests
│   └── main.ipynb             # Interactive notebook
│
├── docs/                        # Documentation Layer
│   ├── COMPREHENSIVE_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   ├── QUICK_START_GUIDE.md
│   └── ARCHITECTURE_DIAGRAM.md
│
├── run_pipeline.py             # Main Execution Script
├── requirements.txt            # Dependencies
└── README.md                   # Project Overview
```

## Technology Stack Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        A1[run_pipeline.py]
        A2[Jupyter Notebooks]
        A3[Test Suite]
    end
    
    subgraph "Business Logic Layer"
        B1[Data Processing]
        B2[Feature Engineering]
        B3[Model Training]
        B4[Evaluation]
    end
    
    subgraph "ML Framework Layer"
        C1[scikit-learn]
        C2[XGBoost]
        C3[imbalanced-learn]
        C4[SDV/TVAE]
    end
    
    subgraph "NLP Processing Layer"
        D1[NLTK]
        D2[WordCloud]
        D3[Text Preprocessing]
    end
    
    subgraph "Data Layer"
        E1[pandas]
        E2[numpy]
        E3[scipy.sparse]
    end
    
    subgraph "Visualization Layer"
        F1[matplotlib]
        F2[seaborn]
    end
    
    subgraph "Infrastructure Layer"
        G1[Python 3.12+]
        G2[File System]
        G3[Memory Management]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> C1
    B2 --> C1
    B3 --> C2
    B4 --> C3
    B1 --> D1
    B2 --> D2
    C1 --> E1
    C2 --> E2
    C3 --> E3
    B4 --> F1
    B4 --> F2
    E1 --> G1
    F1 --> G1
    G1 --> G2
```

## Deployment Architecture Options

### Single Server Deployment
```mermaid
graph TB
    subgraph "Single Server"
        A[Load Balancer/Nginx]
        B[Application Instance]
        C[File Storage]
        D[Model Storage]
    end
    
    E[Users] --> A
    A --> B
    B --> C
    B --> D
```

### Containerized Deployment
```mermaid
graph TB
    subgraph "Docker Environment"
        A[sentiment-analysis:latest]
        B[Volume: /app/datasets]
        C[Volume: /app/models]
        D[Volume: /app/reports]
    end
    
    E[Host System] --> A
    A --> B
    A --> C
    A --> D
```

### Kubernetes Deployment
```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        A[Ingress Controller]
        B[Service]
        C[Deployment]
        D[Pod 1]
        E[Pod 2]
        F[Pod 3]
        G[Persistent Volume]
    end
    
    H[External Traffic] --> A
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    D --> G
    E --> G
    F --> G
```

## Data Processing Pipeline

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw CSV] --> B[Schema Validation]
        B --> C[Data Unification]
    end
    
    subgraph "Text Processing"
        C --> D[Text Cleaning]
        D --> E[Tokenization]
        E --> F[Lemmatization]
        F --> G[Feature Extraction]
    end
    
    subgraph "Feature Engineering"
        G --> H[Train/Test Split]
        H --> I[BoW Vectorization]
        H --> J[TF-IDF Vectorization]
    end
    
    subgraph "Model Training"
        I --> K[Model Training]
        J --> K
        K --> L[Cross Validation]
        L --> M[Model Selection]
    end
    
    subgraph "Data Augmentation"
        I --> N[SMOTE]
        J --> N
        G --> O[TVAE]
        N --> P[Augmented Training]
        O --> P
    end
    
    subgraph "Evaluation"
        M --> Q[Performance Metrics]
        P --> Q
        Q --> R[Visualization]
        Q --> S[Reports]
    end
```

This architecture provides a comprehensive view of how the SocialMedia Sentiment Analysis system is structured, from high-level components down to detailed data flows and deployment options.
