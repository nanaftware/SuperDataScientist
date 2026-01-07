# SuperDataScientist
ensayo que reúne varios modelos para análisis de datos y data science

<img width="1024" height="1024" alt="Gemini_Generated" src="https://github.com/user-attachments/assets/56d38f69-924c-49c6-a334-e1ab19b5ffc0" />

Tiene un instalador automatico con las bibliotecas basicas, y algunas opcionales.

py install_dependencies.py
==  AutoML NLP - Instalador Inteligente  ==

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

===============

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

==  RESUMEN DE INSTALACIÓN  ==

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

==  ⚙️  CONFIGURACIÓN DEL SISTEMA AUTOML  ==

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

🤖 Entrenando modelos de ML... ==
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

==  🏆 SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO  ==

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

==  📋 REPORTE DE CLASIFICACIÓN - MEJOR MODELO  ==
              precision    recall  f1-score   support
    negativo     1.0000    1.0000    1.0000         4
    positivo     1.0000    1.0000    1.0000         4

    accuracy                         1.0000         8
   macro avg     1.0000    1.0000    1.0000         8
weighted avg     1.0000    1.0000    1.0000         8
==


Para datasets mas grades seleccion de dos modelos entre los 16
📝 Iniciando AutoML - Comparador de 2 Modelos
🚀 Versión optimizada para VELOCIDAD


==  ⚙️  CONFIGURACIÓN  ==
¿Usar dataset propio (CSV)? (s/N): s
✅ Cargado: df_limpio.csv (419827 filas)

---📝 MÉTODO DE VECTORIZACIÓN ---
1. TF-IDF (Rápido, basado en palabras exactas)
2. SentenceTransformer (Semántico, más inteligente, requiere más CPU/GPU)

🔽 Selecciona método (1 o 2): 2
   ✅ Usando SentenceTransformer (Embeddings semánticos)

📊 Dataset: 419827 filas
   Columna de texto: 'texto'
   Columna de etiqueta: 'polaridad'

   Distribución de clases:
      positivo: 225279 (53.7%)
      neutro: 133534 (31.8%)
      negativo: 61014 (14.5%)

== 🤖 MODELOS DISPONIBLES EN SDS ==

📌 Selecciona 2 modelos diferentes para comparar


1   Logistic Regression⚡ Modelo lineal rápido y confiable       ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno

2   Ridge Classifier ⚡ Regularización L2, versión lineal robusta ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno

3   SGD Classifier ⚡ Descenso de gradiente estocástico      ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno

4   Multinomial NB 📊 Probabilístico, ideal para conteos de palabras ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno para NLP

5   Bernoulli NB 📊 Probabilístico para características binarias ⚡⚡⚡ Muy rápido  ⭐⭐ Aceptable

6   SVM (Linear) 🎯 Máquinas de soporte vectorial (kernel lineal) ⚡⚡ Rápido       ⭐⭐⭐⭐ Excelente

7   SVM (RBF) 🎯 Máquinas de soporte vectorial (kernel RBF) ⚡ Más lento     ⭐⭐⭐⭐ Muy bueno

8   Decision Tree 🌳 Árbol de decisión simple e interpretable ⚡⚡⚡ Muy rápido  ⭐⭐⭐ Bueno

9   Random Forest 🌲 Ensemble de árboles paralelos          ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno

10  Extra Trees 🌲 Arboles extra aleatorizados (aún más rápido) ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno

11  Gradient Boosting 🚀 Boosting secuencial, excelente precisión ⚡⚡ Rápido       ⭐⭐⭐⭐⭐ Excelente

12  AdaBoost 🚀 Adaptive Boosting, robusto             ⚡⚡ Rápido       ⭐⭐⭐⭐ Muy bueno

13  XGBoost ⚡🚀 Boosting ultra-optimizado, MÁS RÁPIDO ⚡⚡ Rápido       ⭐⭐⭐⭐⭐ Excelente

14  KNN (k=5) 🔍 K-Nearest Neighbors, simple            ⚡ Lento en test ⭐⭐⭐ Bueno

15  LightGBM 💡 Boosting ultra-ligero, MÁS RÁPIDO que XGBoost ⚡⚡⚡ Muy rápido  ⭐⭐⭐⭐⭐ Excelente

16  CatBoost 🐱 Boosting con manejo automático de categorías ⚡⚡ Rápido       ⭐⭐⭐⭐⭐ Excelente


---
💡 RECOMENDACIONES RÁPIDAS:
   - Para MÁXIMA VELOCIDAD: elige 'Logistic Regression' y 'XGBoost'
   - Para MÁXIMA PRECISIÓN: elige 'Gradient Boosting' y 'XGBoost'
   - BALANCEADO: 'Logistic Regression' y 'Random Forest'
---

🔽 Selecciona el MODELO #1 (1-16): 1

   ✅ 'Logistic Regression' seleccionado
      ⚡ Modelo lineal rápido y confiable
      ⚡⚡⚡ Muy rápido | ⭐⭐⭐ Bueno

🔽 Selecciona el MODELO #2 (1-16): 13

   ✅ 'XGBoost' seleccionado
      ⚡🚀 Boosting ultra-optimizado, MÁS RÁPIDO
      ⚡⚡ Rápido | ⭐⭐⭐⭐⭐ Excelente

====
✅ MODELOS SELECCIONADOS
====

1. Logistic Regression
   📝 ⚡ Modelo lineal rápido y confiable
   ⚡ Velocidad: ⚡⚡⚡ Muy rápido
   🎯 Precisión: ⭐⭐⭐ Bueno

2. XGBoost
   📝 ⚡🚀 Boosting ultra-optimizado, MÁS RÁPIDO
   ⚡ Velocidad: ⚡⚡ Rápido
   🎯 Precisión: ⭐⭐⭐⭐⭐ Excelente

⏱️  Tiempo estimado de entrenamiento: ~10 segundos
   (El tiempo real puede variar según el tamaño de tu dataset)

====
⚙️  INICIALIZANDO SISTEMA SDS
====

🔧 Configuración:
   - Lenguaje: Español
   - Test size: 20%
   - Balanceo de clases: SMOTE
   - Vectorización: SENTENCE_TRANSFORMER
   - Hiperparameter tuning: DESACTIVADO (para velocidad)
   - Deep Learning: DESACTIVADO (para velocidad)
   - Modelos a entrenar: ['Logistic Regression', 'XGBoost']

====
🚀 EJECUTANDO PIPELINE
====
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
   - Lematización

   ✓ Preprocesamiento completado
   Longitud promedio original: 267.9 palabras
   Longitud promedio procesado: 133.5 palabras

📊 Analizando frecuencia de palabras...

   Clase 'positivo': 312625 palabras únicas
   Top 10 palabras: [('película', 409784), ('nan', 181300), ('si', 163160), ('historia', 162592), ('cine', 136044), ('ser', 135028), ('bien', 133940), ('mejor', 105250), ('hace', 103013), ('gran', 101998)]

   Clase 'neutro': 242849 palabras únicas
   Top 10 palabras: [('película', 242106), ('si', 116648), ('nan', 108347), ('bien', 90923), ('historia', 87840), ('ser', 82445), ('ver', 63556), ('aunque', 61621), ('final', 59532), ('tan', 59108)]

   Clase 'negativo': 171253 palabras únicas
   Top 10 palabras: [('película', 104189), ('si', 62169), ('nan', 47116), ('ser', 33820), ('ver', 32416), ('tan', 30429), ('historia', 29714), ('cine', 27713), ('bien', 25924), ('hace', 23039)]

📦 Preparando conjuntos de datos...
   Proporción de prueba: 20.0%
   ⚠️  SentenceTransformers no está disponible. Usando TF-IDF por defecto...
   Aplicando vectorización TF-IDF...
   ✓ Conjuntos preparados:
   Entrenamiento: 335861 muestras
   Prueba: 83966 muestras
   Características: 5000 features
   Clases detectadas: ['negativo', 'neutro', 'positivo']

⚖️  Balanceando clases usando: smote
   Distribución original:
   Counter({np.int64(2): 180223, np.int64(1): 106827, np.int64(0): 48811})

   ✓ Clases balanceadas:
   Counter({np.int64(2): 180223, np.int64(1): 180223, np.int64(0): 180223})

🤖 Entrenando modelos de ML...
====
   🎯 Modo SELECTIVO: Entrenando 2 modelo(s) específico(s)
      Modelos seleccionados: ['Logistic Regression', 'XGBoost']
   Total de modelos a entrenar: 2


🔹 Entrenando Logistic Regression...
   Metrics:
   - f1_score: 0.7242
   - balanced_accuracy: 0.7114
   - matthews_corrcoef: 0.5415

🔹 Entrenando XGBoost...
   Metrics:
   - f1_score: 0.6873
   - balanced_accuracy: 0.6308
   - matthews_corrcoef: 0.4721

====
🏆 SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO
====

✨ Mejor modelo seleccionado: Logistic Regression
   Criterio de selección: f1_score

   📊 Métricas del mejor modelo:
   - accuracy: 0.7212
   - precision: 0.7319
   - recall: 0.7212
   - f1_score: 0.7242
   - balanced_accuracy: 0.7114
   - matthews_corrcoef: 0.5415
   - cohen_kappa: 0.5398

   🥇 Ranking de modelos por f1_score:
   1. Logistic Regression: 0.7242
   2. XGBoost: 0.6873

📊 Generando dashboard de resultados...

====
📋 REPORTE DE CLASIFICACIÓN - MEJOR MODELO
====
              precision    recall  f1-score   support

    negativo     0.5727    0.7500    0.6495     12203
      neutro     0.6110    0.5976    0.6042     26707
    positivo     0.8466    0.7866    0.8155     45056

    accuracy                         0.7212     83966
   macro avg     0.6768    0.7114    0.6897     83966
weighted avg     0.7319    0.7212    0.7242     83966


💾 Exportando modelo...
   ✓ Modelo exportado exitosamente: best_model_Logistic_Regression_20260107_012943.joblib
   ✓ Documentación generada: best_model_Logistic_Regression_20260107_012943_README.txt

===
🏆 COMPARATIVA DE LOS 2 MODELOS SELECCIONADOS
===

Modelo                         F1-Score        Balanced Acc    Accuracy
---------------------------------------------------------------------------
Logistic Regression            0.7242          0.7114          0.7212
XGBoost                        0.6873          0.6308          0.6974

===
✨ RESULTADO FINAL
===

🥇 Mejor modelo: Logistic Regression

📊 Métricas del ganador:
   - F1-Score: 0.7242
   - Balanced Accuracy: 0.7114
   - Accuracy: 0.7212
   - Matthews Corrcoef: 0.5415
   - Cohen Kappa: 0.5398

⏱️  Tiempo total de ejecución: 17779.29s
💾 Modelo exportado en: best_model_Logistic_Regression_20260107_012943.joblib

====
📈 ANÁLISIS DE EFICIENCIA
====
✅ Modelos entrenados: 2 de 14+ disponibles
⚡ Tiempo ahorrado: ~120s aproximadamente
💡 Enfoque: Entrenamiento selectivo y eficiente

<img width="1366" height="655" alt="Figure_2" src="https://github.com/user-attachments/assets/afc8fa5e-340e-4cf6-94ff-b76e7cd7f983" />

