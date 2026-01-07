"""
Sistema AutoML de el SDS - Selector de 2 modelos del archivo automl.py
VERSIÓN CORREGIDA: Ahora realmente entrena solo los modelos seleccionados
"""

import pandas as pd
import glob
from automl import AutoNLP, HAS_TENSORFLOW, HAS_TORCH
import time

# ============================================================================
# 1. DICCIONARIO DE MODELOS DISPONIBLES (Extraído de automl.py)
# ============================================================================

AVAILABLE_MODELS = {
    # Modelos lineales
    'Logistic Regression': {
        'description': '⚡ Modelo lineal rápido y confiable',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐ Bueno'
    },
    'Ridge Classifier': {
        'description': '⚡ Regularización L2, versión lineal robusta',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐ Bueno'
    },
    'SGD Classifier': {
        'description': '⚡ Descenso de gradiente estocástico',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐ Bueno'
    },
    
    # Naive Bayes
    'Multinomial NB': {
        'description':  '📊 Probabilístico, ideal para conteos de palabras',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐ Bueno para NLP'
    },
    'Bernoulli NB': {
        'description': '📊 Probabilístico para características binarias',
        'speed':  '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐ Aceptable'
    },
    
    # SVM
    'SVM (Linear)': {
        'description':  '🎯 Máquinas de soporte vectorial (kernel lineal)',
        'speed': '⚡⚡ Rápido',
        'accuracy':  '⭐⭐⭐⭐ Excelente'
    },
    'SVM (RBF)': {
        'description': '🎯 Máquinas de soporte vectorial (kernel RBF)',
        'speed': '⚡ Más lento',
        'accuracy': '⭐⭐⭐⭐ Muy bueno'
    },
    
    # Árboles
    'Decision Tree': {
        'description': '🌳 Árbol de decisión simple e interpretable',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐ Bueno'
    },
    
    # Ensemble - Random Forest
    'Random Forest': {
        'description': '🌲 Ensemble de árboles paralelos',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐ Muy bueno'
    },
    'Extra Trees': {
        'description':  '🌲 Arboles extra aleatorizados (aún más rápido)',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐ Muy bueno'
    },
    
    # Boosting
    'Gradient Boosting': {
        'description': '🚀 Boosting secuencial, excelente precisión',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐⭐ Excelente'
    },
    'AdaBoost': {
        'description': '🚀 Adaptive Boosting, robusto',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐ Muy bueno'
    },
    'XGBoost': {
        'description': '⚡🚀 Boosting ultra-optimizado, MÁS RÁPIDO',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐⭐ Excelente'
    },
    
    # KNN
    'KNN (k=5)': {
        'description': '🔍 K-Nearest Neighbors, simple',
        'speed': '⚡ Lento en test',
        'accuracy': '⭐⭐⭐ Bueno'
    },
    
    # Opcionales (si están instalados)
    'LightGBM': {
        'description': '💡 Boosting ultra-ligero, MÁS RÁPIDO que XGBoost',
        'speed': '⚡⚡⚡ Muy rápido',
        'accuracy': '⭐⭐⭐⭐⭐ Excelente'
    },
    'CatBoost': {
        'description':  '🐱 Boosting con manejo automático de categorías',
        'speed': '⚡⚡ Rápido',
        'accuracy': '⭐⭐⭐⭐⭐ Excelente'
    },
}

# ============================================================================
# 2. FUNCIONES DE INTERFAZ
# ============================================================================

def show_model_menu():
    """
    Menú interactivo para seleccionar 2 modelos
    Muestra info detallada de cada modelo
    """
    model_names = list(AVAILABLE_MODELS.keys())
    
    print("\n" + "="*90)
    print("🤖 MODELOS DISPONIBLES EN SDS")
    print("="*90)
    print("\n📌 Selecciona 2 modelos diferentes para comparar\n")
    
    # Mostrar tabla de modelos
    print(f"{'#':<3} {'Modelo':<25} {'Descripción':<40} {'Velocidad':<15} {'Precisión':<15}")
    print("-" * 98)
    
    for i, name in enumerate(model_names, 1):
        info = AVAILABLE_MODELS[name]
        print(f"{i:<3} {name:<25} {info['description']:<40} {info['speed']:<15} {info['accuracy']:<15}")
    
    print("\n" + "-"*90)
    print("💡 RECOMENDACIONES RÁPIDAS:")
    print("   - Para MÁXIMA VELOCIDAD: elige 'Logistic Regression' y 'XGBoost'")
    print("   - Para MÁXIMA PRECISIÓN: elige 'Gradient Boosting' y 'XGBoost'")
    print("   - BALANCEADO: 'Logistic Regression' y 'Random Forest'")
    print("-"*90)
    
    selected = []
    
    for selection_num in range(1, 3):
        while True:
            try:
                user_input = input(f"\n🔽 Selecciona el MODELO #{selection_num} (1-{len(model_names)}): ").strip()
                
                # Validar entrada
                if not user_input.isdigit():
                    print(f"   ❌ Debes ingresar un número")
                    continue
                
                choice = int(user_input)
                
                if not (1 <= choice <= len(model_names)):
                    print(f"   ❌ Ingresa un número entre 1 y {len(model_names)}")
                    continue
                
                selected_name = model_names[choice - 1]
                
                # Evitar duplicados
                if selected_name in selected:
                    print(f"   ⚠️  '{selected_name}' ya fue seleccionado. Elige OTRO diferente.")
                    continue
                
                info = AVAILABLE_MODELS[selected_name]
                selected.append(selected_name)
                print(f"\n   ✅ '{selected_name}' seleccionado")
                print(f"      {info['description']}")
                print(f"      {info['speed']} | {info['accuracy']}")
                break
                
            except ValueError: 
                print("   ❌ Ingresa un número válido")
            except EOFError:
                # Fallback si no hay entrada interactiva
                print("\n   ⚠️  Modo no-interactivo detectado.")
                print("   Usando modelos por defecto: Logistic Regression + XGBoost")
                return ['Logistic Regression', 'XGBoost']
    
    return selected

def show_selected_models_info(selected_models):
    """Mostrar información detallada de los modelos seleccionados"""
    print("\n" + "="*90)
    print("✅ MODELOS SELECCIONADOS")
    print("="*90)
    
    for i, model_name in enumerate(selected_models, 1):
        info = AVAILABLE_MODELS[model_name]
        print(f"\n{i}. {model_name}")
        print(f"   📝 {info['description']}")
        print(f"   ⚡ Velocidad: {info['speed']}")
        print(f"   🎯 Precisión: {info['accuracy']}")

def estimate_training_time(selected_models):
    """
    Estimar tiempo aproximado de entrenamiento basado en los modelos seleccionados
    """
    # Tiempos base aproximados en segundos para un dataset pequeño
    time_estimates = {
        'Logistic Regression': 2,
        'Ridge Classifier': 2,
        'SGD Classifier': 2,
        'Multinomial NB': 1,
        'Bernoulli NB': 1,
        'SVM (Linear)': 5,
        'SVM (RBF)': 15,
        'Decision Tree': 3,
        'Random Forest': 10,
        'Extra Trees': 10,
        'Gradient Boosting': 20,
        'AdaBoost': 15,
        'XGBoost': 8,
        'KNN (k=5)': 3,
        'LightGBM': 5,
        'CatBoost': 12
    }
    
    total_time = sum(time_estimates.get(model, 10) for model in selected_models)
    
    print(f"\n⏱️  Tiempo estimado de entrenamiento: ~{total_time} segundos")
    print(f"   (El tiempo real puede variar según el tamaño de tu dataset)")

# ============================================================================
# 3. MAIN
# ============================================================================

if __name__ == "__main__": 
    print("🔬 Iniciando SDS - Comparador de 2 Modelos")
    print("🚀 Versión CORREGIDA - Entrena solo los modelos seleccionados\n")
    
    # 1. Cargar datos demo
    data = {
        'texto':  [
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
        ] * 20,
        'polaridad': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo', 'positivo', 'negativo', 'positivo', 'negativo',
                        'positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo'] * 20
    }
    df = pd.DataFrame(data)
    
    # 2. Opción de cargar CSV
    print("\n" + "="*90)
    print("⚙️  CONFIGURACIÓN")
    print("="*90)
    
    try:
        resp = input("¿Usar dataset propio (CSV)? (s/N): ").strip().lower()
    except EOFError:
        resp = 'n'
    
    if resp in ['s', 'si', 'y', 'yes']:
        csv_files = glob.glob('*.csv')
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                print(f"✅ Cargado: {csv_files[0]} ({len(df)} filas)")
            except Exception as e:
                print(f"⚠️  Error: {e}. Usando dataset demo.")
        else:
            print("⚠️  No hay archivos CSV. Usando dataset demo.")
    else:
        print("✅ Usando dataset de demostración")
    
    # 2.2 Opción de Muestreo (Sampling) - CRÍTICO PARA VELOCIDAD
    if len(df) > 50000:
        print(f"\n⚠️  Dataset muy grande detectado ({len(df)} filas).")
        try:
            samp_resp = input("   ¿Deseas usar una muestra para mayor velocidad? (S/n): ").strip().lower()
            if samp_resp != 'n':
                n_samples_str = input("   ¿Cuántas filas usar? (ej. 20000): ").strip()
                n_samples = int(n_samples_str) if n_samples_str else 20000
                df = df.sample(n=min(n_samples, len(df)), random_state=42)
                print(f"   ✅ Dataset reducido a {len(df)} filas")
        except Exception as e:
            print(f"   ⚠️  Error en muestreo: {e}. Usando dataset completo.")
            
    # 2.5 Opción de vectorización
    print("\n" + "-"*40)
    print("📝 MÉTODO DE VECTORIZACIÓN")
    print("-"*40)
    print("1. TF-IDF (Rápido, basado en palabras exactas)")
    print("2. SentenceTransformer (Semántico, más inteligente, requiere más CPU/GPU)")
    
    while True:
        try:
            vec_choice = input("\n🔽 Selecciona método (1 o 2): ").strip()
            if vec_choice == '2':
                vectorization_method = 'sentence_transformer'
                print("   ✅ Usando SentenceTransformer (Embeddings semánticos)")
                break
            else:
                vectorization_method = 'tfidf'
                print("   ✅ Usando TF-IDF (Clásico)")
                break
        except EOFError:
            vectorization_method = 'tfidf'
            break
    
    # Detectar columnas
    text_col = 'texto' if 'texto' in df.columns else df.columns[0]
    label_col = 'polaridad' if 'polaridad' in df.columns else df.columns[1]
    
    print(f"\n📊 Dataset: {len(df)} filas")
    print(f"   Columna de texto: '{text_col}'")
    print(f"   Columna de etiqueta: '{label_col}'")
    print(f"\n   Distribución de clases:")
    for label, count in df[label_col].value_counts().items():
        print(f"      {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 3. Mostrar menú y que el usuario seleccione 2 modelos
    selected_models = show_model_menu()
    show_selected_models_info(selected_models)
    estimate_training_time(selected_models)
    
    # 4. Inicializar AutoML CON LA LISTA DE MODELOS SELECCIONADOS
    print("\n" + "="*90)
    print("⚙️  INICIALIZANDO SISTEMA SDS")
    print("="*90)
    print("\n🔧 Configuración:")
    print("   - Lenguaje: Español")
    print("   - Test size: 20%")
    print("   - Balanceo de clases: OVERSAMPLE (más rápido para datasets grandes)")
    print(f"   - Vectorización: {vectorization_method.upper()}")
    print("   - Hiperparameter tuning: DESACTIVADO (para velocidad)")
    print("   - Deep Learning: DESACTIVADO (para velocidad)")
    print(f"   - Modelos a entrenar: {selected_models}")  # NUEVO: Mostrar qué se va a entrenar
    
    start_time = time.time()
    
    try:
        # MODIFICACIÓN CLAVE: Pasar los modelos seleccionados al constructor
        automl = AutoNLP(
            language='spanish',
            test_size=0.2,
            random_state=42,
            balance_method='oversample', # Cambiado SMOTE -> OVERSAMPLE para velocidad
            custom_metrics=['f1_score', 'balanced_accuracy', 'matthews_corrcoef'],
            use_hyperparameter_tuning=False,
            use_deep_learning=False,
            models_to_train=selected_models,
            vectorization_method=vectorization_method,
            st_model_name='nomic-ai/nomic-embed-text-v1.5', # Opcional: nomic-ai/nomic-embed-text-v1.5
            trust_remote_code=True # Necesario para modelos como Nomic
        )
        
        print("\n" + "="*90)
        print("🚀 EJECUTANDO PIPELINE")
        print("="*90)
        
        # Ejecutar pipeline completo
        best_model, best_model_name = automl.run_full_pipeline(
            df=df,
            text_column=text_col,
            label_column=label_col
        )
        
        # Mostrar comparativa
        print("\n" + "="*90)
        print("🏆 COMPARATIVA DE LOS 2 MODELOS SELECCIONADOS")
        print("="*90)
        
        print(f"\n{'Modelo':<30} {'F1-Score':<15} {'Balanced Acc':<15} {'Accuracy':<15}")
        print("-" * 75)
        
        for model_name in selected_models: 
            if model_name in automl.results:
                metrics = automl.results[model_name]
                print(f"{model_name:<30} {metrics['f1_score']:<15.4f} "
                      f"{metrics['balanced_accuracy']:<15.4f} {metrics['accuracy']:<15.4f}")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*90)
        print(f"✨ RESULTADO FINAL")
        print("="*90)
        print(f"\n🥇 Mejor modelo: {best_model_name}")
        
        if best_model_name in automl.results:
            best_metrics = automl.results[best_model_name]
            print(f"\n📊 Métricas del ganador:")
            print(f"   - F1-Score: {best_metrics['f1_score']:.4f}")
            print(f"   - Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
            print(f"   - Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"   - Matthews Corrcoef: {best_metrics['matthews_corrcoef']:.4f}")
            print(f"   - Cohen Kappa: {best_metrics['cohen_kappa']:.4f}")
        
        print(f"\n⏱️  Tiempo total de ejecución: {elapsed:.2f}s")
        print(f"💾 Modelo exportado en: {automl.model_export_path}")
        
        # Comparación de eficiencia
        print("\n" + "="*90)
        print("📈 ANÁLISIS DE EFICIENCIA")
        print("="*90)
        print(f"✅ Modelos entrenados: {len(selected_models)} de 14+ disponibles")
        print(f"⚡ Tiempo ahorrado: ~{(14 - len(selected_models)) * 10}s aproximadamente")
        print(f"💡 Enfoque: Entrenamiento selectivo y eficiente")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
