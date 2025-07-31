# -*- coding: utf-8 -*-
"""
Модуль для генерации структуры датасета на основе описания пользователя
"""
import re
import time
import torch
from config.prompts import EXTRACT_FIELDS_PROMPT
from utils.field_analyzer import detect_field_type, enhance_field_description


def structure_prompt_v2(
    tokenizer,
    model,
    description: str,
    max_length: int = 4_096,
    debug_mode: bool = False
) -> tuple:
    """
    Создает структуру JSON на основе описания датасета.

    Использует модель для извлечения пар "русское_название:english_key",
    а затем программно строит полную структуру JSON.

    Args:
        tokenizer: Токенизатор модели.
        model: Загруженная модель.
        description: Текстовое описание датасета, предоставленное пользователем.
        max_length: Максимальная длина входа для токенизатора.

    Returns:
        Кортеж (словарь со структурой датасета, количество сгенерированных токенов) или (None, 0) в случае неудачи.
    """
    # Шаг 1: используем модель для извлечения пар "русское_название:english_key"
    prompt = EXTRACT_FIELDS_PROMPT.format(description=description)
    
    if debug_mode:
        print(f"Промпт для создания структуры (длина: {len(prompt)}):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("\n" + "-" * 30)
    
    with torch.inference_mode():
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        if debug_mode:
            print(f"Количество входных токенов для структуры: {inputs['input_ids'].shape[1]}")
        
        generation_start = time.time() if debug_mode else None
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512, #больше как будто не надо, она ломаеться 
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Подсчитываем количество сгенерированных токенов
        input_length = inputs['input_ids'].shape[1]
        total_length = output_ids.shape[1]
        tokens_generated = total_length - input_length
        
        if debug_mode:
            generation_end = time.time()
            print(f"Генерация структуры заняла: {generation_end - generation_start:.2f} сек.")
            print(f"Сгенерировано токенов для структуры: {tokens_generated}")
            print(f"Скорость генерации структуры: {tokens_generated / (generation_end - generation_start):.1f} токенов/сек.")

    # Decode and clean the model's response
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "")
    
    if debug_mode:
        print(f"\nСырой ответ модели для структуры (длина: {len(generated_text)}):")
        print(generated_text[:400] + "..." if len(generated_text) > 400 else generated_text)
        print("\n" + "-" * 30)
    
    field_pairs_str = generated_text.split("Результат:")[-1].strip()
    
    if debug_mode:
        print(f"Извлеченные поля из ответа: {field_pairs_str}")
        print("-" * 30)
    
    if not field_pairs_str or ':' not in field_pairs_str:
        return None, 0

    # Step 2: Parse the response and build the JSON structure in Python
    fields = {}
    example_record = {}
    
    try:
        # Очищаем строку от лишних символов и переносов строк
        field_pairs_str = re.sub(r'[\n\r]+', ' ', field_pairs_str)
        field_pairs_str = re.sub(r'\s+', ' ', field_pairs_str)
        
        # Попробуем несколько разных паттернов для извлечения полей
        patterns_to_try = [
            # Паттерн 1: учитывает запятые внутри примеров - ищет следующее русское поле как разделитель
            r'([а-яА-ЯёЁ\s-]+?)\s*:\s*([a-zA-Z_]+)\s*:\s*(.*?)(?=\s*,\s*[а-яА-ЯёЁ\s-]+?\s*:\s*[a-zA-Z_]+\s*:|$)',
            # Паттерн 2: альтернативный способ с более жадным захватом
            r'([а-яА-ЯёЁ\s-]+?)\s*:\s*([a-zA-Z_]+)\s*:\s*(.*?)(?=,(?=[а-яА-ЯёЁ\s-]+?:)|$)',
            # Паттерн 3: простой split по основным разделителям
            r'([а-яА-ЯёЁ\s-]+?)\s*:\s*([a-zA-Z_]+)\s*:\s*([^,]*?)(?=\s*,\s*[а-яА-ЯёЁ\s-]+?\s*:|$)',
            # Паттерн 4: базовый паттерн
            r'([а-яА-ЯёЁ\s-]+)\s*:\s*([a-zA-Z_]+)\s*:\s*([^,]*)',
        ]
        
        matches = []
        for i, pattern in enumerate(patterns_to_try):
            try:
                current_matches = re.findall(pattern, field_pairs_str, re.IGNORECASE)
                if current_matches:
                    matches = current_matches
                    break
            except Exception:
                continue

        for rus_name, eng_key, example in matches:
            rus_name = rus_name.strip()
            eng_key = eng_key.strip()
            example = example.strip()
            
            # Убираем возможные кавычки и лишние символы из примера
            example = re.sub(r'["\n\r]+.*$', '', example).strip()

            if not rus_name or not eng_key:
                continue

            # Используем пример из описания пользователя, если есть
            if example:
                example_value = example
            else:
                example_value = f'Example for {eng_key}'
            
            # Определяем тип поля на основе контекста
            field_type = detect_field_type(rus_name, example_value, description)
            
            # Обогащаем описание на основе контекста из исходного запроса
            enhanced_description = enhance_field_description(rus_name, example_value, description, field_type)

            fields[eng_key] = {
                'type': field_type,
                'description': enhanced_description,
                'example': example_value
            }
            example_record[eng_key] = example_value

        if not fields:
            return None, 0

        structure = {
            'fields': fields,
            'example_record': example_record
        }
        
        return structure, tokens_generated

    except Exception:
        return None, 0
