"""Environment and Dependency Health Checker.

Verifies Python version, GPU / CUDA availability, scientific packages,
and ensures that all required GIS and ML toolkits are operational.
"""

import sys
import platform
import shutil

print("=" * 65)
print("КОСМОХАКАТОН 2026: ПРОВЕРКА ОКРУЖЕНИЯ И АППАРАТНЫХ РЕСУРСОВ")
print("=" * 65)

# 1. OS & Python
py_ver = platform.python_version()
print(f"Платформа: {platform.system()} {platform.release()} ({platform.machine()})")
print(f"Версия Python: {py_ver}")
if sys.version_info < (3, 10):
    print("  [!] ВНИМАНИЕ: Рекомендуется Python 3.10 или 3.11!")
else:
    print("  [OK] Python соответствует требованиям.")

# 2. PyTorch & GPU
try:
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"  [OK] Доступен GPU: {gpu_name} ({gpu_mem} ГБ VRAM)")
    else:
        print("  [i] GPU не обнаружен (используется CPU-режим).")
        print("      Физическая модель (Track B) полностью оптимизирована для CPU.")
except ImportError:
    print("  [!] PyTorch не установлен (будет работать только Track B).")

# 3. Core Scientific & Geospatial packages
packages = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("sklearn", "Scikit-Learn"),
    ("tifffile", "Tifffile (GeoTIFF I/O)"),
    ("geopandas", "GeoPandas"),
    ("fastapi", "FastAPI (REST API)"),
    ("uvicorn", "Uvicorn (ASGI Server)"),
    ("pytest", "Pytest (Test Engine)"),
]

print("\nПроверка ключевых библиотек:")
all_packages_ok = True
for mod_name, label in packages:
    try:
        mod = __import__(mod_name)
        ver = getattr(mod, "__version__", "OK")
        print(f"  [OK] {label:26} : v{ver}")
    except ImportError:
        print(f"  [X] {label:26} : НЕ НАЙДЕН")
        all_packages_ok = False

# 4. External CLI tools
print("\nПроверка утилит:")
for cli in ["git", "pnpm", "docker"]:
    path = shutil.which(cli)
    status = f"Доступен ({path})" if path else "Не найден (опционально)"
    print(f"  - {cli:8}: {status}")

print("=" * 65)
if all_packages_ok:
    print("ИТОГ: Все необходимые пакеты установлены. Система готова к работе!")
else:
    print("ИТОГ: Обнаружены недостающие пакеты. Выполните: pip install -r requirements.txt")
print("=" * 65)
