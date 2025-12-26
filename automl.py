
"""
Sistema AutoML para Análisis de NLP
Preprocesamiento automático, selección de modelos y visualización de resultados
Optimizado para Google Colab
"""

# ============================================================================
# 1. INSTALACIÓN Y CONFIGURACIÓN DE DEPENDENCIAS
# ============================================================================

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import joblib
import pickle
from collections import Counter
from datetime import datetime
import os
import glob
import io
from PIL import Image

# Configuración de entorno y supresión de avisos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*np.object.*")

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
sns.set_style('whitegrid')

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

# Imbalance & Advanced Models
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from xgboost import XGBClassifier
    from wordcloud import WordCloud
except ImportError:
    print("⚠️ Please restart your kernel. Libraries were installed but haven't been loaded yet.") 

# Deep Learning imports
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
except:
    HAS_TENSORFLOW = False

try:
    from transformers import (BertTokenizer, TFBertForSequenceClassification,
                             DistilBertTokenizer, TFDistilBertForSequenceClassification)
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except:
    HAS_TORCH = False

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except:
    HAS_REPORTLAB = False

# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# 2. CLASE DE PREPROCESAMIENTO DE TEXTO
# ============================================================================

class TextPreprocessor:
    """
    Clase para preprocesamiento automático de texto
    """
    
    def __init__(self, language='spanish'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text):
        """Limpieza básica de texto"""
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = str(text).lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        
        # Eliminar puntuación y caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenización del texto"""
        return word_tokenize(text, language=self.language)
    
    def remove_stopwords(self, tokens):
        """Eliminar stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lematización de tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """Pipeline completo de preprocesamiento"""
        # Limpiar texto
        text = self.clean_text(text)
        
        # Tokenizar
        tokens = self.tokenize(text)
        
        # Eliminar stop words
        tokens = self.remove_stopwords(tokens)
        
        # Lematizar
        tokens = self.lemmatize(tokens)
        
        # Unir tokens de nuevo
        return ' '.join(tokens)

# ============================================================================
# 3. SISTEMA AUTOML PARA NLP
# ============================================================================

class AutoNLP:
    """
    Sistema AutoML para análisis de NLP con selección automática de modelos
    """
    
    def __init__(self, language='spanish', test_size=0.2, random_state=42, 
                 balance_method=None, custom_metrics=None, use_hyperparameter_tuning=False,
                 use_deep_learning=False, max_sequence_length=100):
        self.language = language
        self.test_size = test_size
        self.random_state = random_state
        self.balance_method = balance_method
        self.custom_metrics = custom_metrics or ['accuracy', 'f1_score', 'balanced_accuracy']
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_deep_learning = use_deep_learning
        self.max_sequence_length = max_sequence_length
        self.preprocessor = TextPreprocessor(language=language)
        self.vectorizer = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.word_freq = None
        self.model_export_path = None
        self.tokenizer_dl = None
        self.label_encoder = None
        self.saved_figures = []

    def run_full_pipeline(self, df, text_column, label_column):
        # 1. Cargar y validar los datos
        self.load_data(df, text_column, label_column)
        # 2. Preprocesar los textos
        self.preprocess_data()
        # 3. Analizar frecuencia de palabras (para visualizaciones)
        self.analyze_word_frequency()
        # 4. Preparar conjuntos de entrenamiento y prueba
        self.prepare_datasets()
        # 5. Balancear clases si se especificó
        self.balance_classes()
        # 6. Entrenar los modelos y seleccionar el mejor
        self.train_models()
        # 7. Generar el dashboard con visualizaciones
        self.create_dashboard()
        # 8. Exportar el modelo entrenado
        self.export_model()
        return self.best_model, self.best_model_name

    def predict(self, texts):
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
            
        # Preprocesar
        processed = [self.preprocessor.preprocess(t) for t in texts]
        
        # Vectorizar
        X = self.vectorizer.transform(processed)
        
        # Predecir (numérico)
        y_pred = self.best_model.predict(X)
        
        # Decodificar
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts):
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
            
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError(f"El mejor modelo ({self.best_model_name}) no soporta predict_proba.")
            
        # Preprocesar
        processed = [self.preprocessor.preprocess(t) for t in texts]
        
        # Vectorizar
        X = self.vectorizer.transform(processed)
        
        # Probabilidades
        return self.best_model.predict_proba(X)
        
    def load_data(self, df, text_column, label_column):
        print("📊 Cargando datos...")
        print(f"   Total de registros: {len(df)}")
        print(f"   Columna de texto: '{text_column}'")
        print(f"   Columna de etiquetas: '{label_column}'")
        
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        
        # Verificar valores nulos
        null_count = df[[text_column, label_column]].isnull().sum()
        if null_count.sum() > 0:
            print(f"   ⚠️  Valores nulos encontrados: {null_count.to_dict()}")
            print("   Eliminando filas con valores nulos...")
            self.df = self.df.dropna(subset=[text_column, label_column])
        
        print(f"   ✓ Datos cargados: {len(self.df)} registros válidos")
        print(f"   Distribución de clases:")
        print(self.df[label_column].value_counts())
        
    def preprocess_data(self):
        """Preprocesar todos los textos"""
        print("\n🔧 Preprocesando textos...")
        print("   - Limpieza de texto")
        print("   - Conversión a minúsculas")
        print("   - Tokenización")
        print("   - Eliminación de puntuación")
        print("   - Eliminación de stop words")
        print("   - Lematización")
        
        self.df['processed_text'] = self.df[self.text_column].apply(
            self.preprocessor.preprocess
        )
        
        # Estadísticas de preprocesamiento
        avg_len_original = self.df[self.text_column].str.split().str.len().mean()
        avg_len_processed = self.df['processed_text'].str.split().str.len().mean()
        
        print(f"\n   ✓ Preprocesamiento completado")
        print(f"   Longitud promedio original: {avg_len_original:.1f} palabras")
        print(f"   Longitud promedio procesado: {avg_len_processed:.1f} palabras")
        
    def analyze_word_frequency(self):
        """Analizar palabras más frecuentes por clase"""
        print("\n📊 Analizando frecuencia de palabras...")
        
        self.word_freq = {}
        
        for label in self.df[self.label_column].unique():
            # Filtrar textos por clase
            texts = self.df[self.df[self.label_column] == label]['processed_text']
            
            # Contar palabras
            all_words = ' '.join(texts).split()
            word_counts = Counter(all_words)
            
            # Guardar top palabras
            self.word_freq[label] = dict(word_counts.most_common(50))
            
            print(f"\n   Clase '{label}': {len(word_counts)} palabras únicas")
            print(f"   Top 10 palabras: {list(word_counts.most_common(10))}")
        
        return self.word_freq
    
    def balance_classes(self):
        """Balancear clases del conjunto de entrenamiento"""
        if self.balance_method is None:
            print("\n   ⚖️  Sin balanceo de clases (usando datos originales)")
            return
        
        print(f"\n⚖️  Balanceando clases usando: {self.balance_method}")
        print(f"   Distribución original:")
        print(f"   {Counter(self.y_train)}")
        
        try:
            if self.balance_method == 'oversample':
                ros = RandomOverSampler(random_state=self.random_state)
                self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
                
            elif self.balance_method == 'undersample':
                rus = RandomUnderSampler(random_state=self.random_state)
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
                
            elif self.balance_method == 'smote':
                smote = SMOTE(random_state=self.random_state)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                
            elif self.balance_method == 'smoteenn':
                smoteenn = SMOTEENN(random_state=self.random_state)
                self.X_train, self.y_train = smoteenn.fit_resample(self.X_train, self.y_train)
                
            elif self.balance_method == 'smotetomek':
                smotetomek = SMOTETomek(random_state=self.random_state)
                self.X_train, self.y_train = smotetomek.fit_resample(self.X_train, self.y_train)
            
            print(f"\n   ✓ Clases balanceadas:")
            print(f"   {Counter(self.y_train)}")
            
        except Exception as e:
            print(f"   ⚠️  Error en balanceo: {e}")
            print(f"   Continuando sin balanceo...")

    def prepare_datasets(self):
        """Preparar conjuntos de entrenamiento y prueba"""
        print(f"\n📦 Preparando conjuntos de datos...")
        print(f"   Proporción de prueba: {self.test_size*100}%")
        
        # Codificar etiquetas si no son numéricas
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        
        X = self.df['processed_text']
        y = self.label_encoder.fit_transform(self.df[self.label_column])
        
        # Guardar clases para referencia
        self.classes_ = self.label_encoder.classes_
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Vectorización TF-IDF
        print("   Aplicando vectorización TF-IDF...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        self.X_train = self.vectorizer.fit_transform(X_train)
        self.X_test = self.vectorizer.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"   ✓ Conjuntos preparados:")
        print(f"   Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"   Prueba: {self.X_test.shape[0]} muestras")
        print(f"   Características: {self.X_train.shape[1]} features")
        print(f"   Clases detectadas: {list(self.classes_)}")
        
    def train_models(self):
        """Entrenar múltiples modelos automáticamente"""
        print("\n🤖 Entrenando modelos de ML...")
        print("="*60)
        
        # Importar modelos adicionales
        try:
            from lightgbm import LGBMClassifier
            has_lgbm = True
        except:
            has_lgbm = False
            
        try:
            from catboost import CatBoostClassifier
            has_catboost = True
        except:
            has_catboost = False
        
        # Definir modelos a probar (EXTENDIDO)
        self.models = {
            # Modelos lineales
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Ridge Classifier': RidgeClassifier(random_state=self.random_state),
            'SGD Classifier': SGDClassifier(random_state=self.random_state, loss='log_loss'),
            
            # Naive Bayes
            'Multinomial NB': MultinomialNB(),
            'Bernoulli NB': BernoulliNB(),
            
            # SVM
            'SVM (Linear)': LinearSVC(random_state=self.random_state, max_iter=1000),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            
            # Árboles de decisión
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            
            # Ensemble methods
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state),
            
            # Boosting avanzado
            'XGBoost': XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0),
            
            # KNN
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        }
        
        # Agregar LightGBM si está disponible
        if has_lgbm:
            self.models['LightGBM'] = LGBMClassifier(random_state=self.random_state, verbosity=-1)
        
        # Agregar CatBoost si está disponible
        if has_catboost:
            self.models['CatBoost'] = CatBoostClassifier(random_state=self.random_state, verbose=0)
        
        print(f"   Total de modelos a entrenar: {len(self.models)}")
        
        # Entrenar y evaluar cada modelo
        for name, model in self.models.items():
            print(f"\n🔹 Entrenando {name}...")
            
            try:
                # Hyperparameter tuning si está activado
                if self.use_hyperparameter_tuning and name in self.get_hyperparameter_grids():
                    model = self.tune_hyperparameters(name, model, self.X_train, self.y_train)
                else:
                    # Entrenar modelo normal
                    model.fit(self.X_train, self.y_train)
                
                # Predicciones
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Métricas estándar
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Métricas adicionales
                balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
                mcc = matthews_corrcoef(self.y_test, y_pred)
                kappa = cohen_kappa_score(self.y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                
                # Guardar resultados
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'balanced_accuracy': balanced_acc,
                    'matthews_corrcoef': mcc,
                    'cohen_kappa': kappa,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Mostrar métricas personalizadas
                print(f"   Metrics:")
                for metric in self.custom_metrics:
                    if metric in self.results[name]:
                        print(f"   - {metric}: {self.results[name][metric]:.4f}")
                
            except Exception as e:
                print(f"   ⚠️  Error entrenando {name}: {str(e)}")
                continue
        
        # Entrenar modelos de Deep Learning si está activado
        if self.use_deep_learning:
            self.train_deep_learning_models()
        
        # Seleccionar mejor modelo
        self.select_best_model()

    def get_hyperparameter_grids(self):
        """Obtener grids de hiperparámetros para optimización"""
        grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            },
            'SVM (Linear)': {
                'C': [0.1, 1, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            },
            'Multinomial NB': {
                'alpha': [0.1, 0.5, 1.0]
            }
        }
        return grids

    def tune_hyperparameters(self, name, model, X, y):
        """Optimizar hiperparámetros usando GridSearchCV"""
        grid = self.get_hyperparameter_grids().get(name)
        if not grid:
            return model
            
        print(f"   ⚙️  Optimizando {name}...")
        grid_search = GridSearchCV(
            model, grid, cv=3, scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X, y)
        print(f"      Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_deep_learning_models(self):
        """Entrenar modelos de Deep Learning (LSTM, CNN, BERT)"""
        if not HAS_TENSORFLOW:
            print("\n   ⚠️  TensorFlow no está instalado. Saltando modelos DL.")
            return
        
        print("\n🧠 Entrenando modelos de Deep Learning...")
        print("="*60)
        
        # Preparar datos para DL
        from sklearn.preprocessing import LabelEncoder
        
        # Preparar textos
        texts_train = [' '.join(self.vectorizer.inverse_transform(x)[0]) 
                      for x in self.X_train[:min(1000, self.X_train.shape[0])]]
        texts_test = [' '.join(self.vectorizer.inverse_transform(x)[0]) 
                     for x in self.X_test[:min(1000, self.X_test.shape[0])]]
        
        # Tokenizer para DL
        self.tokenizer_dl = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.tokenizer_dl.fit_on_texts(texts_train)
        
        X_train_seq = self.tokenizer_dl.texts_to_sequences(texts_train)
        X_test_seq = self.tokenizer_dl.texts_to_sequences(texts_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_sequence_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_sequence_length, padding='post')
        
        # Codificar labels (ya codificados en prepare_datasets, pero seleccionamos el rango)
        y_train_enc = self.y_train[:len(texts_train)]
        y_test_enc = self.y_test[:len(texts_test)]
        
        num_classes = len(np.unique(y_train_enc))
        
        # Convertir a one-hot si es necesario
        if num_classes > 2:
            y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test_enc, num_classes)
        else:
            y_train_cat = y_train_enc
            y_test_cat = y_test_enc
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        
        # 1. LSTM Model
        print("\n🔹 Entrenando LSTM...")
        try:
            lstm_model = self._build_lstm_model(num_classes)
            lstm_model.fit(
                X_train_pad, y_train_cat,
                validation_split=0.2,
                epochs=10,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluar
            y_pred_lstm = lstm_model.predict(X_test_pad, verbose=0)
            if num_classes > 2:
                y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
            else:
                y_pred_lstm = (y_pred_lstm > 0.5).astype(int).flatten()
            
            # Métricas
            self._save_dl_results('LSTM', y_test_enc, y_pred_lstm, lstm_model)
            print("   ✓ LSTM entrenado exitosamente")
            
        except Exception as e:
            print(f"   ⚠️  Error en LSTM: {e}")
        
        # 2. CNN Model
        print("\n🔹 Entrenando CNN...")
        try:
            cnn_model = self._build_cnn_model(num_classes)
            cnn_model.fit(
                X_train_pad, y_train_cat,
                validation_split=0.2,
                epochs=10,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluar
            y_pred_cnn = cnn_model.predict(X_test_pad, verbose=0)
            if num_classes > 2:
                y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
            else:
                y_pred_cnn = (y_pred_cnn > 0.5).astype(int).flatten()
            
            # Métricas
            self._save_dl_results('CNN', y_test_enc, y_pred_cnn, cnn_model)
            print("   ✓ CNN entrenado exitosamente")
            
        except Exception as e:
            print(f"   ⚠️  Error en CNN: {e}")
        
        # 3. Bidirectional LSTM
        print("\n🔹 Entrenando Bi-LSTM...")
        try:
            bilstm_model = self._build_bilstm_model(num_classes)
            bilstm_model.fit(
                X_train_pad, y_train_cat,
                validation_split=0.2,
                epochs=10,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluar
            y_pred_bilstm = bilstm_model.predict(X_test_pad, verbose=0)
            if num_classes > 2:
                y_pred_bilstm = np.argmax(y_pred_bilstm, axis=1)
            else:
                y_pred_bilstm = (y_pred_bilstm > 0.5).astype(int).flatten()
            
            # Métricas
            self._save_dl_results('Bi-LSTM', y_test_enc, y_pred_bilstm, bilstm_model)
            print("   ✓ Bi-LSTM entrenado exitosamente")
            
        except Exception as e:
            print(f"   ⚠️  Error en Bi-LSTM: {e}")
    
    def _build_lstm_model(self, num_classes):
        """Construir modelo LSTM"""
        model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=self.max_sequence_length),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes if num_classes > 2 else 1, 
                 activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_cnn_model(self, num_classes):
        """Construir modelo CNN"""
        model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=self.max_sequence_length),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(pool_size=4),
            Conv1D(64, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes if num_classes > 2 else 1,
                 activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_bilstm_model(self, num_classes):
        """Construir modelo Bidirectional LSTM"""
        model = Sequential([
            Embedding(input_dim=5000, output_dim=128, input_length=self.max_sequence_length),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes if num_classes > 2 else 1,
                 activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_dl_results(self, model_name, y_true, y_pred, model):
        """Guardar resultados de modelos DL"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        self.results[model_name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': None,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'matthews_corrcoef': mcc,
            'cohen_kappa': kappa,
            'cv_mean': accuracy,  # No hay CV para DL
            'cv_std': 0
        }
        
    def select_best_model(self):
        """Seleccionar automáticamente el mejor modelo"""
        print("\n" + "="*60)
        print("🏆 SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO")
        print("="*60)
        
        # Determinar métrica principal para selección
        main_metric = self.custom_metrics[0] if self.custom_metrics else 'f1_score'
        
        # Ordenar por métrica principal
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1][main_metric], 
            reverse=True
        )
        
        self.best_model_name = sorted_models[0][0]
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n✨ Mejor modelo seleccionado: {self.best_model_name}")
        print(f"   Criterio de selección: {main_metric}")
        print(f"\n   📊 Métricas del mejor modelo:")
        
        # Mostrar todas las métricas disponibles
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 
                          'balanced_accuracy', 'matthews_corrcoef', 'cohen_kappa']
        
        for metric in metrics_to_show:
            if metric in self.results[self.best_model_name]:
                value = self.results[self.best_model_name][metric]
                print(f"   - {metric}: {value:.4f}")
        
        # Mostrar top 5 modelos
        print(f"\n   🥇 Top 5 modelos por {main_metric}:")
        for i, (name, result) in enumerate(sorted_models[:5], 1):
            print(f"   {i}. {name}: {result[main_metric]:.4f}")
        
    def export_model(self, filename=None, export_format='joblib'):
        print("\n💾 Exportando modelo...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_{self.best_model_name.replace(' ', '_')}_{timestamp}"
        
        # Crear diccionario con todo lo necesario
        model_package = {
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'label_column': self.label_column,
            'text_column': self.text_column,
            'classes': list(self.classes_),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            if export_format == 'joblib':
                filepath = f"{filename}.joblib"
                joblib.dump(model_package, filepath)
            else:  # pickle
                filepath = f"{filename}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(model_package, f)
            
            self.model_export_path = filepath
            print(f"   ✓ Modelo exportado exitosamente: {filepath}")
            print(f"   Formato: {export_format}")
            print(f"   Tamaño: {self._get_file_size(filepath)}")
            
            # Crear archivo README
            readme_path = f"{filename}_README.txt"
            self._create_readme(readme_path, model_package)
            print(f"   ✓ Documentación creada: {readme_path}")
            
            return filepath
            
        except Exception as e:
            print(f"   ❌ Error exportando modelo: {str(e)}")
            return None
    
    def _get_file_size(self, filepath):
        """Obtener tamaño del archivo"""
        import os
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        else:
            return f"{size_bytes/(1024**2):.2f} MB"
    
    def _create_readme(self, filepath, model_package):
        """Crear archivo README con información del modelo"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MODELO DE NLP - DOCUMENTACIÓN\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Modelo: {model_package['model_name']}\n")
            f.write(f"Fecha de entrenamiento: {model_package['training_date']}\n")
            f.write(f"Clases: {', '.join(map(str, model_package['classes']))}\n\n")
            
            f.write("MÉTRICAS DE RENDIMIENTO:\n")
            f.write("-" * 40 + "\n")
            metrics = model_package['metrics']
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']:
                if key in metrics:
                    f.write(f"{key}: {metrics[key]:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("CÓMO CARGAR Y USAR EL MODELO:\n")
            f.write("="*60 + "\n\n")
            
            f.write("import joblib\n\n")
            f.write("# Cargar modelo\n")
            f.write(f"model_package = joblib.load('{self.model_export_path}')\n")
            f.write("model = model_package['model']\n")
            f.write("vectorizer = model_package['vectorizer']\n")
            f.write("preprocessor = model_package['preprocessor']\n\n")
            
            f.write("# Hacer predicción\n")
            f.write("texto = 'Tu texto aquí'\n")
            f.write("texto_procesado = preprocessor.preprocess(texto)\n")
            f.write("texto_vectorizado = vectorizer.transform([texto_procesado])\n")
            f.write("prediccion = model.predict(texto_vectorizado)\n")
            f.write("probabilidad = model.predict_proba(texto_vectorizado)\n")
    
    @staticmethod
    def load_model(filepath):
        print(f"\n📂 Cargando modelo desde: {filepath}")
        
        try:
            if filepath.endswith('.joblib'):
                model_package = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model_package = pickle.load(f)
            
            print(f"   ✓ Modelo cargado: {model_package['model_name']}")
            print(f"   Entrenado el: {model_package['training_date']}")
            print(f"   Clases: {', '.join(map(str, model_package['classes']))}")
            
            return model_package
            
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {str(e)}")
            return None
    def create_dashboard(self):
        """Crear dashboard completo con visualizaciones"""
        print("\n📊 Generando dashboard de resultados...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Comparación de modelos
        ax1 = plt.subplot(2, 3, 1)
        self._plot_model_comparison(ax1)
        
        # 2. Matriz de confusión del mejor modelo
        ax2 = plt.subplot(2, 3, 2)
        self._plot_confusion_matrix(ax2)
        
        # 3. Métricas detalladas
        ax3 = plt.subplot(2, 3, 3)
        self._plot_detailed_metrics(ax3)
        
        # 4. ROC Curve (si es binario)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_roc_curve(ax4)
        
        # 5. Predicciones vs Reales
        ax5 = plt.subplot(2, 3, 5)
        self._plot_predictions_vs_actual(ax5)
        
        # 6. Distribución de confianza
        ax6 = plt.subplot(2, 3, 6)
        self._plot_confidence_distribution(ax6)
        
        plt.tight_layout()
        plt.show()
        
        # Reporte de clasificación
        print("\n" + "="*60)
        print("📋 REPORTE DE CLASIFICACIÓN - MEJOR MODELO")
        print("="*60)
        print(classification_report(
            self.y_test, 
            self.results[self.best_model_name]['predictions'],
            target_names=self.classes_,
            digits=4
        ))
        
    def _plot_model_comparison(self, ax):
        """Gráfico de comparación entre modelos"""
        # Usar métricas personalizadas
        metrics = self.custom_metrics[:4] if len(self.custom_metrics) >= 4 else self.custom_metrics
        model_names = list(self.results.keys())
        
        # Limitar número de modelos si son muchos
        if len(model_names) > 10:
            model_names = model_names[:10]
            ax.set_title('Top 10 Modelos - Comparación', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Comparación de Modelos', fontsize=14, fontweight='bold')
        
        x = np.arange(len(model_names))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = [self.results[name].get(metric, 0) for name in model_names]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Modelos', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics)-1) / 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
    def _plot_confusion_matrix(self, ax):
        """Matriz de confusión del mejor modelo"""
        cm = confusion_matrix(self.y_test, self.results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                    xticklabels=self.classes_, yticklabels=self.classes_)
        ax.set_title(f'Matriz de Confusión - {self.best_model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor Real', fontsize=12)
        ax.set_xlabel('Predicción', fontsize=12)
        
    def _plot_detailed_metrics(self, ax):
        """Métricas detalladas del mejor modelo"""
        best_results = self.results[self.best_model_name]
        
        # Incluir métricas avanzadas
        metrics = {
            'Accuracy': best_results.get('accuracy', 0),
            'Precision': best_results.get('precision', 0),
            'Recall': best_results.get('recall', 0),
            'F1-Score': best_results.get('f1_score', 0),
            'Balanced Acc': best_results.get('balanced_accuracy', 0),
            'MCC': best_results.get('matthews_corrcoef', 0),
            'Cohen Kappa': best_results.get('cohen_kappa', 0)
        }
        
        # Filtrar métricas que existen
        metrics = {k: v for k, v in metrics.items() if v != 0}
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
        bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.8)
        
        ax.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Métricas Detalladas - {self.best_model_name}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores en las barras
        for bar, value in zip(bars, metrics.values()):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', va='center', fontweight='bold', fontsize=9)
        
    def _plot_roc_curve(self, ax):
        """Curva ROC (solo para clasificación binaria)"""
        if len(np.unique(self.y_test)) == 2:
            proba = self.results[self.best_model_name]['probabilities']
            if proba is not None:
                fpr, tpr, _ = roc_curve(self.y_test, proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
                ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
                ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Probabilidades no disponibles', 
                       ha='center', va='center', fontsize=12)
                ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'ROC solo para clasificación binaria', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
        
    def _plot_predictions_vs_actual(self, ax):
        """Comparación de predicciones vs valores reales"""
        predictions = self.results[self.best_model_name]['predictions']
        
        # Contar coincidencias
        correct = (predictions == self.y_test).sum()
        incorrect = len(self.y_test) - correct
        
        sizes = [correct, incorrect]
        labels = [f'Correctas\n({correct})', f'Incorrectas\n({incorrect})']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        
    def _plot_confidence_distribution(self, ax):
        """Distribución de confianza en las predicciones"""
        proba = self.results[self.best_model_name]['probabilities']
        
        if proba is not None:
            max_proba = np.max(proba, axis=1)
            
            ax.hist(max_proba, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(max_proba.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Media: {max_proba.mean():.3f}')
            ax.set_xlabel('Confianza de Predicción', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Confianza', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Probabilidades no disponibles', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Distribución de Confianza', fontsize=14, fontweight='bold')
    
    def _plot_top_words_per_class(self, ax):
        """Gráfico de palabras más frecuentes por clase"""
        if self.word_freq is None:
            ax.text(0.5, 0.5, 'Ejecuta analyze_word_frequency() primero', 
                   ha='center', va='center', fontsize=11)
            ax.set_title('Top Palabras por Clase', fontsize=12, fontweight='bold')
            return
        
        # Tomar primera clase
        first_class = list(self.word_freq.keys())[0]
        top_words = dict(list(self.word_freq[first_class].items())[:15])
        
        words = list(top_words.keys())
        counts = list(top_words.values())
        
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(words)))
        ax.barh(words, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Frecuencia', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 15 Palabras - Clase: {first_class}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_wordcloud(self, ax, class_idx=0):
        """Generar WordCloud para una clase"""
        if self.word_freq is None:
            ax.text(0.5, 0.5, 'Ejecuta analyze_word_frequency() primero', 
                   ha='center', va='center', fontsize=11)
            ax.set_title('WordCloud', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        classes = list(self.word_freq.keys())
        if class_idx >= len(classes):
            class_idx = 0
        
        class_name = classes[class_idx]
        word_freq = self.word_freq[class_name]
        
        try:
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(word_freq)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'WordCloud - Clase: {class_name}', fontsize=12, fontweight='bold')
            ax.axis('off')
        except:
            ax.text(0.5, 0.5, 'Error generando WordCloud', 
                   ha='center', va='center', fontsize=11)
            ax.axis('off')
    
    def _plot_advanced_metrics_comparison(self, ax):
        """Comparar métricas avanzadas entre modelos"""
        model_names = list(self.results.keys())[:8]  # Top 8 modelos
        
        metrics = ['balanced_accuracy', 'matthews_corrcoef', 'cohen_kappa']
        metric_labels = ['Balanced Acc', 'MCC', 'Cohen Kappa']
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[name].get(metric, 0) for name in model_names]
            ax.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Modelos', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Métricas Avanzadas', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_cv_scores(self, ax):
        """Gráfico de scores de cross-validation"""
        model_names = list(self.results.keys())[:8]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(model_names)))
        
        ax.bar(x, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xlabel('Modelos', fontsize=11, fontweight='bold')
        ax.set_ylabel('CV Score', fontsize=11, fontweight='bold')
        ax.set_title('Cross-Validation Scores (5-fold)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
    
    def _plot_error_analysis(self, ax):
        """Análisis de errores del mejor modelo"""
        predictions = self.results[self.best_model_name]['predictions']
        
        # Crear matriz de errores
        n_classes = len(self.classes_)
        
        if n_classes <= 2:
            # Para clasificación binaria
            correct = (predictions == self.y_test).sum()
            errors = len(self.y_test) - correct
            
            data = [correct, errors]
            labels = ['Correctas', 'Errores']
            colors = ['#2ecc71', '#e74c3c']
            
            wedges, texts, autotexts = ax.pie(data, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            # Para multiclase, mostrar errores por clase
            errors_per_class = []
            for i in range(n_classes):
                mask = self.y_test == i
                class_errors = (predictions[mask] != i).sum()
                errors_per_class.append(class_errors)
            
            ax.bar(range(n_classes), errors_per_class, color='salmon', alpha=0.8)
            ax.set_xlabel('Clases', fontsize=11, fontweight='bold')
            ax.set_ylabel('Número de Errores', fontsize=11, fontweight='bold')
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(self.classes_, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        ax.set_title('Análisis de Errores', fontsize=12, fontweight='bold')
    
    def _plot_class_distribution(self, ax):
        """Distribución de clases en train y test"""
        train_dist = Counter(self.y_train)
        test_dist = Counter(self.y_test)
        
        indices = range(len(self.classes_))
        train_counts = [train_dist.get(i, 0) for i in indices]
        test_counts = [test_dist.get(i, 0) for i in indices]
        
        x = np.arange(len(self.classes_))
        width = 0.35
        
        ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, test_counts, width, label='Test', alpha=0.8, color='coral')
        
        ax.set_xlabel('Clases', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cantidad', fontsize=11, fontweight='bold')
        ax.set_title('Distribución de Clases', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes_, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')