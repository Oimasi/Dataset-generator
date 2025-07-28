import os
import json
import torch
from model_utils import load_model, generate_chunk, structure_prompt_v2

MODEL_CHOICES = [
    ("GPT-2 Small (124M)", "gpt2"),
    ("DistilGPT-2 (82M)", "distilgpt2"),
    ("TinyLlama 1.1B Chat v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Mistral Small 3.1 (24B)", "mistralai/Mistral-7B-Instruct-v0.1"),
    ("Qwen3-4B", "Qwen/Qwen3-4B")
]

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
    print(f"\n=== Генератор датасетов (консоль) ===\nИспользуем устройство: {device.upper()}\n")

    # Выбор модели
    model_id = choose_model("Выберите модель (одна и та же будет использована для структуры и генерации):")

    # Токен
    token = input("Введите ваш Hugging Face токен (или оставьте пустым): ").strip()
    if token:
        os.environ['HUGGINGFACE_TOKEN'] = token

    # Описание датасета
    print("\nВведите описание датасета. Для завершения введите пустую строку.")
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
    default_path = os.getcwd()
    out = input(f"Путь для сохранения файла ({default_path}): ").strip() or default_path
    if not os.path.isdir(out):
        try:
            os.makedirs(out)
        except Exception as e:
            print(f"Не удалось создать директорию: {e}")
            return

    print("\nЗагрузка модели...")
    tokenizer, model = load_model(model_id, device=device, token=token)

    print("Формируем шаблон структуры...")
    prompt = structure_prompt_v2(tokenizer, model, description, device=device)
    print(f"Сформированный шаблон:\n{prompt}\n")

    # Генерация чанков
    dataset = []
    for i in range(1, n_chunks + 1):
        print(f"Генерация чанка {i}/{n_chunks}...")
        chunk = generate_chunk(tokenizer, model, prompt)
        dataset.append(chunk)

    # Сохранение
    output_file = os.path.join(out, "synthetic_dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Датасет сохранен: {output_file}\n")

if __name__ == '__main__':
    main()
