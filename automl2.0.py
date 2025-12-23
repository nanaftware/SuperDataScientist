"""
Sistema AutoML para Análisis de NLP
Preprocesamiento automático, selección de modelos y visualización de resultados
Compatible con Windows, Linux y macOS

INSTALACIÓN RECOMENDADA (Windows):
pip install nltk scikit-learn pandas numpy matplotlib seaborn xgboost wordcloud imbalanced-learn lightgbm catboost reportlab pillow

OPCIONAL (Deep Learning):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
"""

# ============================================================================
# 1. INSTALACIÓN Y CONFIGURACIÓN DE DEPENDENCIAS
# ============================================================================

# Instalar dependencias necesarias (VERSIÓN COMPATIBLE)
# Para Windows y Python 3.8+
#pip install nltk scikit-learn pandas numpy matplotlib seaborn xgboost wordcloud imbalanced-learn lightgbm catboost reportlab pillow


print("✅ Dependencias básicas instaladas correctamente")

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from collections import Counter

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Importaciones de sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, balanced_accuracy_score)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from xgboost import XGBClassifier
import joblib
import pickle
from datetime import datetime
from wordcloud import WordCloud
import io
from PIL import Image

# Deep Learning imports (OPCIONAL - el sistema funciona sin estos)
HAS_TENSORFLOW = False
HAS_TRANSFORMERS = False
HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Dense, LSTM, Embedding, Dropout, 
                                         Bidirectional, GlobalMaxPooling1D,
                                         Conv1D, MaxPooling1D, Flatten)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
    print("✅ TensorFlow cargado correctamente")
except ImportError:
    print("⚠️  TensorFlow no disponible - Modelos DL desactivados")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    print("✅ PyTorch cargado correctamente")
except ImportError:
    print("⚠️  PyTorch no disponible")

try:
    from transformers import (BertTokenizer, BertForSequenceClassification,
                             DistilBertTokenizer, DistilBertForSequenceClassification)
    HAS_TRANSFORMERS = True
    print("✅ Transformers cargado correctamente")
except ImportError:
    print("⚠️  Transformers no disponible")

# PDF generation imports
HAS_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
    print("✅ ReportLab cargado correctamente")
except ImportError:
    print("⚠️  ReportLab no disponible - PDFs desactivados")

# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# 2. IMPORTAR LIBRERÍA LOCAL
# ============================================================================

from automl import AutoNLP

# ============================================================================
# 4. EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Crear datos de ejemplo (reemplazar con tus propios datos)
    print("📝 Creando datos de ejemplo...")
    print("   (Reemplaza esto con tu propio dataset)")

    # Ejemplo: Dataset de sentimientos de reseñas
    data = {
        'texto': [
            'Este producto es excelente, me encantó la calidad',
            'Muy malo, no lo recomiendo para nada',
            'Increíble servicio, volveré a comprar',
            'Pésima experiencia, nunca más',
            'Buena relación calidad-precio',
            'No cumple las expectativas',
            'Fantástico, superó mis expectativas',
            'Decepcionante, esperaba más',
            'Muy bueno, lo recomiendo ampliamente',
            'Terrible, una pérdida de dinero',
            'Excelente atención al cliente',
            'Mala calidad, se rompió rápido',
            'Perfecto para lo que necesitaba',
            'No vale la pena, muy caro',
            'Maravilloso producto, cinco estrellas',
            'Horrible experiencia de compra'
        ] * 20,  # Multiplicamos para tener más datos
        'sentimiento': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo', 'positivo', 'negativo', 'positivo', 'negativo',
                        'positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo'] * 20
    }

    df = pd.DataFrame(data)

    print(f"✓ Dataset creado: {len(df)} registros")
    print("\n💡 Para usar tus propios datos:")
    print("   df = pd.read_csv('tu_archivo.csv')")
    print("   o")
    print("   df = pd.read_excel('tu_archivo.xlsx')")

    # ============================================================================
    # 5. EJECUTAR PIPELINE AUTOML CON CONFIGURACIÓN AVANZADA
    # ============================================================================
    print("\n" + "="*60)
    print("⚙️  CONFIGURACIÓN DEL SISTEMA AUTOML")
    print("="*60)
    print("""
    Características Disponibles:
    ✅ 16+ Modelos de Machine Learning
    ✅ Hyperparameter Tuning Automático (GridSearchCV)
    ✅ 5 Métodos de Balanceo de Clases
    ✅ 7 Métricas Avanzadas
    ✅ 12 Visualizaciones
    ✅ Exportación Automática (PNG/PDF)
    ✅ Reporte PDF Completo
    ✅ Análisis de Palabras Frecuentes
    ✅ WordClouds por Clase

    Características Opcionales (requieren instalación adicional):
    ⚠️  3 Modelos de Deep Learning (LSTM, CNN, Bi-LSTM) - Requiere PyTorch/TensorFlow
    """)

    # Verificar disponibilidad de Deep Learning
    dl_available = HAS_TENSORFLOW or HAS_TORCH
    if dl_available:
        print("✅ Deep Learning disponible")
    else:
        print("ℹ️  Deep Learning no disponible (instala PyTorch o TensorFlow para activarlo)")

    # Inicializar sistema AutoML con opciones avanzadas
    automl = AutoNLP(
        language='spanish',                    # 'spanish' o 'english'
        test_size=0.2,                        # Proporción de datos para test
        random_state=42,                      # Semilla para reproducibilidad
        balance_method='smote',               # None, 'oversample', 'undersample', 'smote', 'smoteenn', 'smotetomek'
        custom_metrics=['f1_score', 'balanced_accuracy', 'matthews_corrcoef'],  # Métricas principales
        use_hyperparameter_tuning=True,      # Activar GridSearchCV (más lento pero mejor)
        use_deep_learning=False,             # ⚠️ Cambiar a True si tienes PyTorch/TensorFlow instalado
        max_sequence_length=100               # Longitud máxima para DL
    )

    print("\n💡 Configuración seleccionada:")
    print(f"   - Balanceo de clases: {automl.balance_method}")
    print(f"   - Hyperparameter tuning: {automl.use_hyperparameter_tuning}")
    print(f"   - Deep Learning: {automl.use_deep_learning}")
    print(f"   - Métricas: {', '.join(automl.custom_metrics)}")

    # Ejecutar pipeline completo
    best_model, best_model_name = automl.run_full_pipeline(
        df=df,
        text_column='texto',
        label_column='sentimiento'
    )