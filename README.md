# SuperDataScientist
ensayo que reúne varios modelos para análisis de datos y data science

Tiene un instalador automatico con las bibliotecas basicas, y algunas opcionales.

py install_dependencies.py
============================================================
  AutoML NLP - Instalador Inteligente
============================================================

🔍 Verificando dependencias...

✅ nltk
✅ scikit-learn
✅ pandas
✅ numpy
✅ matplotlib
✅ seaborn
✅ xgboost
✅ wordcloud
✅ imbalanced-learn
✅ lightgbm
✅ catboost
✅ reportlab
✅ pillow
✅ joblib
⚠️  pytorch (opcional)
✅ transformers
✅ tensorflow
✅ keras

============================================================

✅ Todos los paquetes esenciales ya están instalados

💡 Hay 1 paquetes opcionales disponibles:
   - pytorch

¿Deseas instalar los paquetes opcionales? (s/n): s

📦 Instalando paquetes opcionales...

▶ pytorch...    Instalando pytorch...


📚 Descargando recursos de NLTK...
   ✅ punkt
   ✅ stopwords
   ✅ wordnet
   ✅ averaged_perceptron_tagger

============================================================
  RESUMEN DE INSTALACIÓN
============================================================

✅ Paquetes instalados: 18/18

🎉 ¡Instalación completada exitosamente!

Ahora puedes ejecutar el script AutoML:
---python automl2.0.py---
   
✅ Detecta automáticamente qué librerías están instaladas
✅ Muestre mensajes claros de qué falta y cómo instalarlo
✅ Desactive automáticamente funcionalidades que requieren librerías faltantes

py automl2.0.py

1-Carga y valida los datos.
2-Preprocesa los textos.
3-Analiza la frecuencia de palabras (para visualizaciones).
4-Prepara los conjuntos de entrenamiento y prueba.
5-Balancea las clases (si se especificó).
6-Entrena los modelos y selecciona el mejor.
7-Genera el dashboard con todas las visualizaciones.
8-Exporta el modelo entrenado.

============================================================
⚙️  CONFIGURACIÓN DEL SISTEMA AUTOML
============================================================

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

✅ Deep Learning disponible

💡 Configuración seleccionada:
   - Balanceo de clases: smote
   - Hyperparameter tuning: True
   - Deep Learning: False
   - Métricas: f1_score, balanced_accuracy, matthews_corrcoef
 Cargando datos...
   Total de registros: 40
   Columna de texto: 'texto'
   Columna de etiquetas: 'sentimiento'
   ✓ Datos cargados: 40 registros válidos
   Distribución de clases:
sentimiento
positivo    20
negativo    20
Name: count, dtype: int64

🔧 Preprocesando textos...
   - Limpieza de texto
   - Conversión a minúsculas
   - Tokenización
   - Eliminación de puntuación
   - Eliminación de stop words
   - Lematización

   ✓ Preprocesamiento completado
   Longitud promedio original: 1.0 palabras
   Longitud promedio procesado: 1.0 palabras

📦 Preparando conjuntos de datos...
   Proporción de prueba: 20.0%
   Aplicando vectorización TF-IDF...
   ✓ Conjuntos preparados:
   Entrenamiento: 32 muestras
   Prueba: 8 muestras
   Características: 2 features
   Clases detectadas: ['negativo', 'positivo']

🤖 Entrenando modelos de ML...
============================================================
   Total de modelos a entrenar: 16

🔹 Entrenando Logistic Regression...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Ridge Classifier...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando SGD Classifier...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Multinomial NB...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Bernoulli NB...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando SVM (Linear)...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando SVM (RBF)...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Decision Tree...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Random Forest...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Extra Trees...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando Gradient Boosting...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando AdaBoost...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando XGBoost...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando KNN (k=5)...
   Metrics:
   - accuracy: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000

🔹 Entrenando LightGBM...
   Metrics:
   - accuracy: 0.5000
   - f1_score: 0.3333
   - balanced_accuracy: 0.5000

🔹 Entrenando CatBoost...
   ⚠️  Error entrenando CatBoost: The following error was raised: 'CatBoostClassifiier' object has no attribute '__sklearn_tags__'. It seems that there are no classes that implement `__sklearn_tags__` in the MRO and/or all classes in the MRO call `super().__sklearn_tags__()`. Make sure to inherit from `BaseEstimator` which implements `__sklearn_tags__` (or alternatively define `__sklearn_tags__` but we don't recommend this approach). Note that `BaseEstimator` needs to be on the right side of other Mixins in the inheritance order.

============================================================
🏆 SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO
============================================================

✨ Mejor modelo seleccionado: Logistic Regression
   Criterio de selección: accuracy

   📊 Métricas del mejor modelo:
   - accuracy: 1.0000
   - precision: 1.0000
   - recall: 1.0000
   - f1_score: 1.0000
   - balanced_accuracy: 1.0000
   - matthews_corrcoef: 1.0000
   - cohen_kappa: 1.0000

   🥇 Top 5 modelos por accuracy:
   1. Logistic Regression: 1.0000
   2. Ridge Classifier: 1.0000
   3. SGD Classifier: 1.0000
   4. Multinomial NB: 1.0000
   5. Bernoulli NB: 1.0000
✓ train_models completed

📊 Generando dashboard de resultados...

<img width="1366" height="655" alt="DashBoard" src="https://github.com/user-attachments/assets/e1874b73-e541-4b8d-a5e6-aa6871b2c999" />

============================================================
📋 REPORTE DE CLASIFICACIÓN - MEJOR MODELO
============================================================
              precision    recall  f1-score   support
    negativo     1.0000    1.0000    1.0000         4
    positivo     1.0000    1.0000    1.0000         4

    accuracy                         1.0000         8
   macro avg     1.0000    1.0000    1.0000         8
weighted avg     1.0000    1.0000    1.0000         8
============================================================


Para datasets mas grades seleccion de dos modelos entre los 16
📝 Iniciando AutoML - Comparador de 2 Modelos
🚀 Versión optimizada para VELOCIDAD


==========================================================================================
⚙️  CONFIGURACIÓN
==========================================================================================
¿Usar dataset propio (CSV)? (s/N): s
✅ Cargado:  df_limpio.csv (419827 filas)

📊 Dataset: 419827 filas
   Columna de texto: 'texto'
   Columna de etiqueta: 'polaridad'

   Distribución de clases:
      positivo: 225279 (53.7%)
      neutro: 133534 (31.8%)
      negativo: 61014 (14.5%)

==========================================================================================
🤖 MODELOS DISPONIBLES EN AutoML
==========================================================================================

📌 Selecciona 2 modelos diferentes para comparar

#   Modelo                    Descripción                              Velocidad       Precisión    

--------------------------------------------------------------------------------------------------  
1   Logistic Regression       ⚡ Modelo lineal rápido y confiable       ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno
2   Ridge Classifier          ⚡ Regularización L2, versión lineal robusta ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno
3   SGD Classifier            ⚡ Descenso de gradiente estocástico      ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno
4   Multinomial NB            📊 Probabilístico, ideal para conteos de palabras ⚡⚡⚡ Muy rápido   ⭐⭐⭐ Bueno para NLP
5   Bernoulli NB              📊 Probabilístico para características binarias ⚡⚡⚡ Muy rápido  ⭐ ⭐ Aceptable
6   SVM (Linear)              🎯 Máquinas de soporte vectorial (kernel lineal) ⚡⚡ Rápido       ⭐ ⭐⭐⭐ Excelente
7   SVM (RBF)                 🎯 Máquinas de soporte vectorial (kernel RBF) ⚡ Más lento     ⭐⭐⭐ ⭐ Muy bueno
8   Decision Tree             🌳 Árbol de decisión simple e interpretable ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno
9   Random Forest             🌲 Ensemble de árboles paralelos          ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno
10  Extra Trees               🌲 Arboles extra aleatorizados (aún más rápido) ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno
11  Gradient Boosting         🚀 Boosting secuencial, excelente precisión ⚡⚡ Rápido       ⭐⭐⭐⭐⭐ Excelente
12  AdaBoost                  🚀 Adaptive Boosting, robusto             ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno
13  XGBoost                   ⚡🚀 Boosting ultra-optimizado, MÁS RÁPIDO ⚡⚡ Rápido       ⭐⭐⭐⭐ ⭐ Excelente
14  KNN (k=5)                 📍 K-Nearest Neighbors, simple            ⚡ Lento en test ⭐⭐⭐ Bueno
15  LightGBM                  💡 Boosting ultra-ligero, MÁS RÁPIDO que XGBoost ⚡⚡⚡ Muy rápido  ⭐⭐⭐⭐⭐ Excelente
16  CatBoost                  🐱 Boosting con manejo automático de categorías ⚡⚡ Rápido       ⭐⭐⭐⭐⭐ Excelente

------------------------------------------------------------------------------------------
💡 RECOMENDACIONES RÁPIDAS:
   - Para MÁXIMA VELOCIDAD: elige 'Logistic Regression' y 'XGBoost'
   - Para MÁXIMA PRECISIÓN: elige 'Gradient Boosting' y 'XGBoost'
   - BALANCEADO: 'Logistic Regression' y 'Random Forest'
------------------------------------------------------------------------------------------

🔽 Selecciona el MODELO #1 (1-16): 1

   ✅ 'Logistic Regression' seleccionado
      ⚡ Modelo lineal rápido y confiable
      ⚡⚡⚡ Muy rápido | ⭐⭐⭐ Bueno

🔽 Selecciona el MODELO #2 (1-16): 12

   ✅ 'AdaBoost' seleccionado
      🚀 Adaptive Boosting, robusto
      ⚡⚡ Rápido | ⭐⭐⭐⭐ Muy bueno

==========================================================================================
✅ MODELOS SELECCIONADOS
==========================================================================================

1. Logistic Regression
   📝 ⚡ Modelo lineal rápido y confiable
   ⚡ Velocidad: ⚡⚡⚡ Muy rápido
   🎯 Precisión: ⭐⭐⭐ Bueno

2. AdaBoost
   📝 🚀 Adaptive Boosting, robusto
   ⚡ Velocidad: ⚡⚡ Rápido
   🎯 Precisión: ⭐⭐⭐⭐ Muy bueno

==========================================================================================
⚙️  INICIALIZANDO SISTEMA AUTOML
==========================================================================================

🔧 Configuración:
   - Lenguaje:  Español
   - Test size: 20%
   - Balanceo de clases:  SMOTE
   - Hiperparameter tuning:  DESACTIVADO (para velocidad)
   - Deep Learning:  DESACTIVADO (para velocidad)

==========================================================================================
🚀 EJECUTANDO PIPELINE
==========================================================================================
📊 Cargando datos...
   Total de registros: 419827
   Columna de texto: 'texto'
   Columna de etiquetas: 'polaridad'
   ✓ Datos cargados: 419827 registros válidos
   Distribución de clases:
polaridad
positivo    225279
neutro      133534
negativo     61014
Name: count, dtype: int64

🔧 Preprocesando textos...
   - Limpieza de texto
   - Conversión a minúsculas
   - Tokenización
   - Eliminación de puntuación
   - Eliminación de stop words
   - Lematización...
