============================================================
MODELO DE NLP - DOCUMENTACIÓN
============================================================

Modelo: Logistic Regression
Fecha de entrenamiento: 2025-12-23 12:21:55
Clases: negativo, positivo

MÉTRICAS DE RENDIMIENTO:
----------------------------------------
accuracy: 1.0000
precision: 1.0000
recall: 1.0000
f1_score: 1.0000
balanced_accuracy: 1.0000

============================================================
CÓMO CARGAR Y USAR EL MODELO:
============================================================

import joblib

# Cargar modelo
model_package = joblib.load('best_model_Logistic_Regression_20251223_122155.joblib')
model = model_package['model']
vectorizer = model_package['vectorizer']
preprocessor = model_package['preprocessor']

# Hacer predicción
texto = 'Tu texto aquí'
texto_procesado = preprocessor.preprocess(texto)
texto_vectorizado = vectorizer.transform([texto_procesado])
prediccion = model.predict(texto_vectorizado)
probabilidad = model.predict_proba(texto_vectorizado)
