import os
import json
import torch
import base64
import warnings

# Отключаем все предупреждения, они тока мешают и ничего не делают 
warnings.filterwarnings('ignore')

# Отключаем предупреждения TensorFlow (должно быть установлено до импорта transformers)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from model_utils import load_model, generate_chunk, structure_prompt_v2

ENV_FILE = ".env"
TOKEN_KEY = "HF_TOKEN"

MODEL_CHOICES = [
    ("Qwen3-4B", "Qwen/Qwen3-4B")
]

def load_env():
    """Загружает переменные из .env файла"""
    env_vars = {}
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except Exception:
            pass
    return env_vars

def save_env(key, value):
    """Сохраняет переменную в .env файл с простым кодированием"""
    try:
        # Кодируем токен в base64 для минимальной обфускации
        encoded_value = base64.b64encode(value.encode()).decode()
        
        env_vars = load_env()
        env_vars[key] = encoded_value
        
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.write("# Файл переменных окружения\n")
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        return True
    except Exception:
        return False

def load_token():
    """Загружает токен из .env файла"""
    env_vars = load_env()
    if TOKEN_KEY in env_vars:
        try:
            # Декодируем токен из base64
            return base64.b64decode(env_vars[TOKEN_KEY]).decode()
        except Exception:
            return None
    return None

def save_token(token):
    """Сохраняет токен в .env файл""" #да,я написал функцию для одной строчки код, и че вы мне сделаете
    return save_env(TOKEN_KEY, token)

def get_token():
    """Получает токен: сначала пытается загрузить из файла, если нет - запрашивает у пользователя"""
    token = load_token()
    if token:
        print(f"Найден сохранённый токен: {token[:8]}...")
        use_saved = input("Использовать сохранённый токен? (y/n): ").strip().lower()
        if use_saved in ['y', 'yes', 'да', 'д', '']:
            return token
    
    # Запрашиваем новый токен
    token = input("Введите ваш Hugging Face токен (или оставьте пустым): ").strip()
    if token:
        save_token(token)
        print("Токен сохранён для будущих запусков.")
    return token

def choose_model(prompt_text):
    print(prompt_text)
    for idx, (name, _) in enumerate(MODEL_CHOICES, 1):
        print(f"  {idx}. {name}")
    while True:
        choice = input("Введите номер модели: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(MODEL_CHOICES):
            return MODEL_CHOICES[int(choice)-1][1]
        print("Некорректный выбор. Попробуйте снова.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Генератор датасетов  ===\nИспользуем устройство: {device.upper()}\n")

    # Выбор модели
    model_id = choose_model("Выберите модель (одна и та же будет использована для структуры и генерации):")

    # Токен
    token = get_token()
    if token:
        os.environ['HUGGINGFACE_TOKEN'] = token

    # Описание датасета
    print("\nВведите описание датасета. Для завершения нажмите Enter 2 раза")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    description = "\n".join(lines).strip()
    if not description:
        print("Ошибка: описание не может быть пустым.")
        return

    # Количество чанков
    while True:
        n_chunks = input("Количество чанков для генерации (по умолчанию 10): ").strip() or "10"
        if n_chunks.isdigit() and int(n_chunks) > 0:
            n_chunks = int(n_chunks)
            break
        print("Введите положительное целое число.")

    # Путь сохранения
    out = os.getcwd()

    print("\nЗагрузка модели...")
    tokenizer, model = load_model(model_id, device=device, token=token)

    print("Формируем шаблон структуры...")
    structure = None
    for attempt in range(3): # 3 попытки на создание структуры
        structure = structure_prompt_v2(tokenizer, model, description)
        if structure:
            break
        print(f"Попытка {attempt + 1} не удалась, пробую снова...")

    if not structure:
        print("Не удалось создать структуру датасета после нескольких попыток. Завершение работы.")
        return
    print(f"Сформированная структура:\n{json.dumps(structure, ensure_ascii=False, indent=2)}\n")

    # Генерация и запись чанков в файл
    dataset = []
    output_file = os.path.join(out, "synthetic_dataset.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n') # Открываем JSON-массив
        
        for i in range(1, n_chunks + 1):
            print(f"Генерация чанка {i}/{n_chunks}...")
            
            # Повторяем генерацию, пока не получим валидный и уникальный чанк
            while True:
                chunk = generate_chunk(tokenizer, model, structure, description)
                
                # Проверяем, что чанк сгенерирован и не является дубликатом
                if chunk and chunk not in dataset:
                    dataset.append(chunk) # Для проверки дубликатов
                    
                    # Добавляем запятую перед новым элементом (кроме первого)
                    if i > 1:
                        f.write(',\n')
                    
                    # Записываем чанк в файл
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                    f.flush() # Принудительно записываем данные на диск
                    break
                elif chunk in dataset:
                    print(f"Обнаружен дубликат для чанка {i}. Повторная генерация...")
                # Если chunk равен None, цикл просто продолжится для новой попытки
        
        f.write('\n]') # Закрываем JSON-массив

    print(f"\nГотово! Датасет сохранен: {output_file}\n")

if __name__ == '__main__':
    main()
