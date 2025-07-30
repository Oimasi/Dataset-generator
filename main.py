import os
import json
import torch
import base64
import warnings

# Отключаем все предупреждения - они только засоряют вывод
warnings.filterwarnings('ignore')

# Отключаем предупреждения TensorFlow - нужно сделать до импорта transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from model_utils import load_model, generate_chunk, structure_prompt_v2

ENV_FILE = ".env"
TOKEN_KEY = "HF_TOKEN"

MODEL_CHOICES = [
    ("Qwen3-4B", "Qwen/Qwen3-4B"),
    ("Qwen3-8B", "Qwen/Qwen3-8B")
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
        # Кодируем токен в base64 чтобы не светился в открытом виде
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
            # Декодируем токен обратно
            return base64.b64decode(env_vars[TOKEN_KEY]).decode()
        except Exception:
            return None
    return None

def save_token(token):
    """Сохраняет токен в .env файл."""
    return save_env(TOKEN_KEY, token)

def get_token():
    """Получает токен: сначала пытается загрузить из файла, если не удается, запрашивает у пользователя."""
    token = load_token()
    if token:
        use_saved = input(f"Найден сохраненный токен: {token[:8]}... Использовать его? (y/n): ").strip().lower()
        if use_saved in ['y', 'yes', '']:
            return token
    
    # Запрашиваем новый токен
    token = input("Введите ваш токен Hugging Face (или оставьте пустым): ").strip()
    if token:
        save_token(token)
        print("Токен сохранен для будущих запусков.")
    return token

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Выбор модели
    print("\nДоступные модели:")
    for i, (name, model_id) in enumerate(MODEL_CHOICES, 1):
        print(f"{i}. {name} ({model_id})")
    
    while True:
        choice = input(f"\nВыберите модель (1-{len(MODEL_CHOICES)}) или нажмите Enter для модели по умолчанию ({MODEL_CHOICES[0][0]}): ").strip()
        
        if not choice:  # Выбираем по умолчанию
            model_id = MODEL_CHOICES[0][1]
            print(f"Выбрана модель по умолчанию: {MODEL_CHOICES[0][0]}")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(MODEL_CHOICES):
            selected_idx = int(choice) - 1
            model_id = MODEL_CHOICES[selected_idx][1]
            print(f"Выбрана модель: {MODEL_CHOICES[selected_idx][0]}")
            break
        else:
            print(f"Пожалуйста, введите число от 1 до {len(MODEL_CHOICES)} или нажмите Enter.")

    # Получаем токен
    token = get_token()
    if token:
        os.environ['HUGGINGFACE_TOKEN'] = token

    # Описание датасета
    print("\nВведите описание датасета (нажмите Enter дважды для завершения):")
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

    # Количество частей
    while True:
        n_chunks = input("Количество частей для генерации (по умолчанию 10): ").strip() or "10"
        if n_chunks.isdigit() and int(n_chunks) > 0:
            n_chunks = int(n_chunks)
            break
        print("Пожалуйста, введите положительное целое число.")

    # Путь для сохранения
    out = os.getcwd()

    print("Загрузка модели...")
    tokenizer, model = load_model(model_id, device=device, token=token)

    print("Создание шаблона структуры...")
    structure = None
    for attempt in range(3): # Пробуем создать структуру несколько раз
        structure = structure_prompt_v2(tokenizer, model, description)
        if structure:
            print("\nСтруктура сгенерирована успешно:")
            print(json.dumps(structure, ensure_ascii=False, indent=2))
            break

    if not structure:
        print("Не удалось создать структуру датасета после нескольких попыток. Выход.")
        return

    # Генерируем части и записываем в файл
    dataset = []
    output_file = os.path.join(out, "synthetic_dataset.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n') # Открываем JSON массив
        
        for i in range(1, n_chunks + 1):
            print(f"Генерация части {i}/{n_chunks}...")
            
            # Повторяем генерацию пока не получим уникальную запись
            while True:
                chunk = generate_chunk(tokenizer, model, structure, description)
                
                if chunk and chunk not in dataset:
                    dataset.append(chunk) # Для проверки дубликатов
                    
                    if i > 1:
                        f.write(',\n')
                    
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                    f.flush() # Принудительно записываем на диск
                    break
                elif chunk in dataset:
                    print(f"Обнаружен дубликат для части {i}. Повторная генерация...")
        
        f.write('\n]') # Закрываем JSON массив

    print(f"\nГотово! Датасет сохранен в: {output_file}\n")

if __name__ == '__main__':
    main()
