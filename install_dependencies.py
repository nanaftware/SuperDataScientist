"""
Script de instalación inteligente para AutoML NLP
Detecta qué falta e instala solo lo necesario
"""

import subprocess
import sys

def check_package(package_name, import_name=None):
    """Verificar si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Instalar un paquete con pip"""
    try:
        print(f"   Instalando {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        return True
    except:
        return False

def main():
    print("="*60)
    print("  AutoML NLP - Instalador Inteligente")
    print("="*60)
    print()
    
    # Lista de paquetes necesarios
    packages = {
        # (nombre_pip, nombre_import, esencial)
        'nltk': ('nltk', 'nltk', True),
        'scikit-learn': ('scikit-learn', 'sklearn', True),
        'pandas': ('pandas', 'pandas', True),
        'numpy': ('numpy', 'numpy', True),
        'matplotlib': ('matplotlib', 'matplotlib', True),
        'seaborn': ('seaborn', 'seaborn', True),
        'xgboost': ('xgboost', 'xgboost', False),
        'wordcloud': ('wordcloud', 'wordcloud', False),
        'imbalanced-learn': ('imbalanced-learn', 'imblearn', True),
        'lightgbm': ('lightgbm', 'lightgbm', False),
        'catboost': ('catboost', 'catboost', False),
        'reportlab': ('reportlab', 'reportlab', False),
        'pillow': ('pillow', 'PIL', True),
        'joblib': ('joblib', 'joblib', True),
        'pytorch': ('pytorch', 'torch', False),
        'transformers': ('transformers', 'transformers', False),
        'tensorflow': ('tensorflow', 'tensorflow', False),
        'keras': ('keras', 'keras', False),
    }
    
    print("🔍 Verificando dependencias...\n")
    
    missing_essential = []
    missing_optional = []
    installed = []
    
    for pip_name, (pkg_pip, pkg_import, essential) in packages.items():
        if check_package(pkg_pip, pkg_import):
            installed.append(pip_name)
            print(f"✅ {pip_name}")
        else:
            if essential:
                missing_essential.append(pkg_pip)
                print(f"❌ {pip_name} (ESENCIAL)")
            else:
                missing_optional.append(pkg_pip)
                print(f"⚠️  {pip_name} (opcional)")
    
    print("\n" + "="*60)
    
    # Instalar paquetes faltantes esenciales
    if missing_essential:
        print(f"\n📦 Instalando {len(missing_essential)} paquetes esenciales...\n")
        
        for package in missing_essential:
            print(f"▶ {package}...", end=" ")
            if install_package(package):
                print("✅")
            else:
                print("❌ ERROR")
                print(f"\n⚠️  No se pudo instalar {package}")
                print(f"   Intenta manualmente: pip install {package}\n")
    else:
        print("\n✅ Todos los paquetes esenciales ya están instalados")
    
    # Preguntar por paquetes opcionales
    if missing_optional:
        print(f"\n💡 Hay {len(missing_optional)} paquetes opcionales disponibles:")
        for pkg in missing_optional:
            print(f"   - {pkg}")
        
        print("\n¿Deseas instalar los paquetes opcionales? (s/n): ", end="")
        
        try:
            response = input().lower()
            if response in ['s', 'si', 'y', 'yes']:
                print("\n📦 Instalando paquetes opcionales...\n")
                for package in missing_optional:
                    print(f"▶ {package}...", end=" ")
                    if install_package(package):
                        print("✅")
                    else:
                        print("❌ (no crítico)")
        except:
            print("\nSaltando instalación de paquetes opcionales")
    
    # Descargar recursos de NLTK
    print("\n📚 Descargando recursos de NLTK...")
    try:
        import nltk
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                print(f"   ✅ {resource}")
            except:
                print(f"   ⚠️  {resource} (no crítico)")
    except:
        print("   ⚠️  NLTK no disponible")
    
    # Resumen final
    print("\n" + "="*60)
    print("  RESUMEN DE INSTALACIÓN")
    print("="*60)
    
    total = len(packages)
    
    # Verificar de nuevo después de instalar
    now_installed = sum(1 for pip_name, (pkg_pip, pkg_import, _) in packages.items() 
                       if check_package(pkg_pip, pkg_import))
    
    print(f"\n✅ Paquetes instalados: {now_installed}/{total}")
    
    if now_installed >= total - len(missing_optional):
        print("\n🎉 ¡Instalación completada exitosamente!")
        print("\nAhora puedes ejecutar el script AutoML:")
        print("   python automl.py")
    else:
        print("\n⚠️  Algunas dependencias esenciales faltan")
        print("\nIntenta instalar manualmente:")
        for pkg in missing_essential:
            if not check_package(pkg):
                print(f"   pip install {pkg}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()