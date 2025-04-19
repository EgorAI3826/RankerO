from huggingface_hub import snapshot_download
import os

# Параметры
repo_id = "Egor-AI/RankerO_Datsets"
script_dir = os.path.dirname(os.path.abspath(__file__))  # Директория, где находится .py файл
local_dir = os.path.join(script_dir, "dataset")  # Папка dataset в директории скрипта

# Создание директории dataset, если она не существует
os.makedirs(local_dir, exist_ok=True)

# Скачивание всех JSON-файлов из репозитория
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns="*.json"  # Скачиваем только JSON-файлы
)

print(f"Все JSON-файлы скачаны в {local_dir}")