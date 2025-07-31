import os
import json
import torch
import warnings
import time

# Отключаем все предупреждения - они только засоряют вывод
warnings.filterwarnings('ignore')

# Отключаем предупреждения TensorFlow - нужно сделать до импорта transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импорты из новой модульной структуры
from core.model_loader import load_model
from generators.structure_generator import structure_prompt_v2
from generators.data_generator import generate_chunk
from utils.token_manager import get_token
from utils.token_calculator import calculate_max_tokens
from config.model_config import MODEL_CHOICES


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Выбор модели
    print("\nДоступные модели:")
    for i, (name, model_id) in enumerate(MODEL_CHOICES, 1):
        print(f"{i}. {name} ({model_id})")
    
    debug_mode = False
    while True:
        choice = input(f"\nВыберите модель (1-{len(MODEL_CHOICES)}) или нажмите Enter для модели по умолчанию ({MODEL_CHOICES[0][0]}): ").strip()

        if choice.startswith('-') and choice[1:].isdigit() and 1 <= abs(int(choice)) <= len(MODEL_CHOICES):  # Дебаг режим
            debug_mode = True
            selected_idx = abs(int(choice)) - 1
            model_id = MODEL_CHOICES[selected_idx][1]
            print(f"Выбрана модель в дебаг режиме: {MODEL_CHOICES[selected_idx][0]}")
            break
        elif not choice:  # Выбираем по умолчанию
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
    description = input("Введите описание датасета: ").strip()
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

    if debug_mode:
        print(f"\n=== ДЕБАГ РЕЖИМ АКТИВИРОВАН ===")
        print(f"Устройство: {device}")
        print(f"Модель: {model_id}")
        print(f"Описание датасета: {description}")
        print(f"Количество частей: {n_chunks}")
        print(f"Начало работы: {time.ctime()}")
        print("=" * 50)
    
    load_start = time.time()
    tokenizer, model = load_model(model_id, device=device, token=token)
    load_time = time.time() - load_start
    
    if debug_mode:
        print(f"Модель загружена за: {load_time:.2f} сек.")
        print(f"Тип модели: {type(model).__name__}")
        print(f"Тип токенизатора: {type(tokenizer).__name__}")
        if hasattr(model, 'config'):
            print(f"Конфигурация модели: {model.config.model_type if hasattr(model.config, 'model_type') else 'Неизвестно'}")
        print("=" * 50)

    print("Создание шаблона структуры...")
    structure = None
    for attempt in range(3): # Пробуем создать структуру несколько раз
        start_time = time.time()
        if debug_mode: print(f"Начало попытки {attempt + 1} в {time.ctime(start_time)}")
        print(f"Попытка {attempt + 1}/3: анализ описания и генерация структуры...")

        structure, tokens_generated = structure_prompt_v2(tokenizer, model, description, debug_mode=debug_mode)

        if structure:
            generation_time = time.time() - start_time
            print(f"✓ Структура сгенерирована успешно! (сгенерировано {tokens_generated} токенов за {generation_time:.1f} сек.)")
            if debug_mode:
                print("\nДетали структуры:")
                print(json.dumps(structure, ensure_ascii=False, indent=2))
                print(f"Генерация завершилась в {time.ctime(time.time())}")
            print("\nПоля датасета:")
            for key, value in structure['fields'].items():
                print(f"  {key} - {value['type']}")
            break
        else:
            generation_time = time.time() - start_time
            print(f"✗ Попытка {attempt + 1} неудачна ({generation_time:.1f} сек.)")

    # Рассчитываем оптимальное количество токенов на основе описания
    optimal_tokens = calculate_max_tokens(description, base_tokens=512, min_tokens=256, max_tokens=2048)
    
    if debug_mode:
        print(f"\nОптимальное количество токенов для генерации: {optimal_tokens}")
        print(f"Длина описания: {len(description)} символов")
        print("=" * 50)
    else:
        print(f"Рассчитано {optimal_tokens} токенов для каждой записи")
    
    # Генерируем части и записываем в файл
    dataset = []
    output_file = os.path.join(out, "synthetic_dataset.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n') # Открываем JSON массив
        
        for i in range(1, n_chunks + 1):
            start_time = time.time()
            print(f"Генерация части {i}/{n_chunks} ({i/n_chunks*100:.1f}%)...")
            if debug_mode:
                print(f"Начало генерации части {i} в {time.ctime(start_time)}")

            # Повторяем генерацию пока не получим уникальную запись
            while True:
                chunk = generate_chunk(tokenizer, model, structure, description, max_new_tokens=optimal_tokens, debug_mode=debug_mode)

                if chunk and chunk not in dataset:
                    dataset.append(chunk) # Для проверки дубликатов

                    if i > 1:
                        f.write(',\n')

                    json.dump(chunk, f, ensure_ascii=False, indent=2)
                    f.flush() # Принудительно записываем на диск
                    end_time = time.time()
                    if debug_mode:
                        print(json.dumps(chunk, ensure_ascii=False, indent=2))
                        print(f"Запись сгенерирована за {end_time - start_time:.2f} сек. Завершено в {time.ctime(end_time)}")
                    print(f"Успешно создана запись с полями: {list(chunk.keys())} ({round(time.time() - start_time, 2)} сек.)")
                    break
                elif chunk in dataset:
                    if debug_mode:
                        print(f"Дубликат обнаружен: {json.dumps(chunk, ensure_ascii=False)}")
                    print(f"Обнаружен дубликат для части {i}. Повторная генерация...")
                else:
                    if debug_mode:
                        print(f"Ошибка генерации для части {i}. Повторная попытка...")
                        if chunk is None:
                            print("Получен None от generate_chunk")
                        else:
                            print(f"Получено: {chunk}")
        
        f.write('\n]') # Закрываем JSON массив

    total_time = time.time() - (load_start if 'load_start' in locals() else time.time())
    if debug_mode:
        print("\n" + "=" * 50)
        print("ИТОГОВАЯ СТАТИСТИКА:")
        print(f"Общее время работы: {total_time:.2f} сек.")
        print(f"Сгенерировано записей: {len(dataset)}")
        print(f"Среднее время на запись: {total_time/len(dataset):.2f} сек." if len(dataset) > 0 else "")
        print(f"Размер файла: {os.path.getsize(output_file) / 1024:.2f} KB")
        print(f"Завершено: {time.ctime()}")
        print("=" * 50)
    print(f"\nГотово! Датасет сохранен в: {output_file}\n")

if __name__ == '__main__':
    main()
