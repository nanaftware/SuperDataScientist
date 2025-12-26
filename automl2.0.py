"""
Sistema AutoML para Análisis de NLP
Preprocesamiento automático, selección de modelos y visualización de resultados
Compatible con Windows, Linux y macOS
"""

import pandas as pd
import glob
from automl import AutoNLP, HAS_TENSORFLOW, HAS_TORCH


# ============================================================================
# 4. EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("📝 Iniciando sistema AutoML...")

    # 1. Preparar datos de demo
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
        ] * 20,
        'polaridad': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo', 'positivo', 'negativo', 'positivo', 'negativo',
                        'positivo', 'negativo', 'positivo', 'negativo', 'positivo', 
                        'negativo'] * 20
    }
    df = pd.DataFrame(data)

    print("\n" + "="*60)
    print("⚙️  CONFIGURACIÓN DEL SISTEMA AUTOML")
    print("="*60)

    # Pregunta interactiva para usar dataset propio
    try:
        respuesta = input("¿Quieres analizar un dataset propio (CSV)? (s/N): ").strip().lower()
    except EOFError:
        respuesta = 'n'

    if respuesta in ['s', 'si', 'y', 'yes']:
        csv_files = glob.glob('*.csv')
        if csv_files:
            archivo_csv = csv_files[0]
            print(f"📁 Encontrado archivo CSV: {archivo_csv}")
            try:
                df_custom = pd.read_csv(archivo_csv)
                if not df_custom.empty:
                    df = df_custom
                    print(f"✅ Dataset '{archivo_csv}' cargado con {len(df)} registros.")
                else:
                    print("⚠️  El archivo CSV está vacío. Usando demo.")
            except Exception as e:
                print(f"❌ Error cargando CSV: {e}. Usando demo.")
        else:
            print("⚠️  No se encontraron archivos CSV en el directorio. Usando demo.")
    else:
        print("✅ Usando dataset de demo.")

    # Determinar columnas automáticamente si es posible
    text_col = 'texto' if 'texto' in df.columns else df.columns[0]
    label_col = 'polaridad' if 'polaridad' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    print(f"🎯 Resumen del Dataset:")
    print(f"   - Registros: {len(df)}")
    print(f"   - Columna de texto: {text_col}")
    print(f"   - Columna de etiquetas: {label_col}")
    
    if label_col in df.columns:
        print(f"   - Distribución de clases:\n{df[label_col].value_counts()}")

    # 2. Inicializar sistema AutoML
    automl = AutoNLP(
        language='spanish',
        test_size=0.2,
        random_state=42,
        balance_method='smote',
        custom_metrics=['f1_score', 'balanced_accuracy', 'matthews_corrcoef'],
        use_hyperparameter_tuning=True,#Para ganar tiempo en un ordenador menos potente podemos cambiar esto a =False
        use_deep_learning=False
    )

    # 3. Ejecutar pipeline completo
    try:
        best_model, best_model_name = automl.run_full_pipeline(
            df=df,
            text_column=text_col,
            label_column=label_col
        )
        print(f"\n✨ ¡Proceso completado! Mejor modelo: {best_model_name}")
    except Exception as e:

        print(f"\n❌ Error durante la ejecución del pipeline: {e}")
