# -*- coding: utf-8 -*-
"""
Модуль для генерации записей данных на основе предоставленной структуры
"""
import json
import time
import torch
from config.prompts import DATA_GENERATION_PROMPT, CONTEXTUAL_DATA_GENERATION_PROMPT
from utils.json_handler import safe_json_parse, validate_and_fix_json_structure
from utils.field_analyzer import convert_field_types


def generate_chunk(
    tokenizer,
    model,
    structure,
    description: str = "",
    max_new_tokens: int = 256,
    debug_mode: bool = False
) -> dict:
    """
    Генерирует одну запись данных (чанк) на основе предоставленной структуры.

    Args:
        tokenizer: Токенизатор модели.
        model: Загруженная модель.
        structure (dict): Словарь, описывающий структуру данных.
        description (str): Общее описание датасета для контекста.
        max_new_tokens (int): Максимальное количество новых токенов для генерации.

    Returns:
        Словарь с сгенерированными данными или None, если генерация или разбор не удались.
    """
    if not isinstance(structure, dict) or 'fields' not in structure:
        return {"error": "Неверный формат структуры"}
    
    # ПОЛУЧАЕМ СТРОГИЙ СПИСОК ОБЯЗАТЕЛЬНЫХ КЛЮЧЕЙ
    required_keys = list(structure['fields'].keys())
    required_keys_str = ', '.join([f'"{key}"' for key in required_keys])
    
    # Создаем описания полей для промпта
    field_descriptions = []
    for field_name, field_info in structure['fields'].items():
        field_descriptions.append(
            f"- {field_name} ({field_info['type']}): {field_info['description']}"
        )
    
    example_record_str = json.dumps(structure['example_record'], ensure_ascii=False, indent=2)
    
    # Используем контекстный промпт, если есть описание
    if description.strip():
        generation_prompt = CONTEXTUAL_DATA_GENERATION_PROMPT.format(
            description=description,
            field_descriptions=chr(10).join(field_descriptions),
            example_record=example_record_str,
            required_keys=required_keys_str
        )
    else:
        generation_prompt = DATA_GENERATION_PROMPT.format(
            description=description,
            field_descriptions=chr(10).join(field_descriptions),
            example_record=example_record_str,
            required_keys=required_keys_str
        )

    with torch.inference_mode():
        inputs = tokenizer(
            generation_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        if debug_mode:
            print(f"Промпт для генерации (длина: {len(generation_prompt)}):") 
            print(generation_prompt[:500] + "..." if len(generation_prompt) > 500 else generation_prompt)
            print(f"Количество входных токенов: {inputs['input_ids'].shape[1]}")
        
        generation_start = time.time() if debug_mode else None
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,  # Больше креативности для данных
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        if debug_mode:
            generation_end = time.time()
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = output_ids.shape[1] - input_tokens
            print(f"Генерация заняла: {generation_end - generation_start:.2f} сек.")
            print(f"Сгенерировано токенов: {output_tokens}")
            print(f"Скорость генерации: {output_tokens / (generation_end - generation_start):.1f} токенов/сек.")
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Убираем исходный промпт
    if generation_prompt in generated_text:
        generated_text = generated_text.replace(generation_prompt, "").strip()
    
    if debug_mode:
        print(f"Сырой текст от модели (длина: {len(generated_text)}):")
        print(generated_text[:300] + "..." if len(generated_text) > 300 else generated_text)
    
    # Пытаемся извлечь JSON с помощью улучшенного парсера
    result = safe_json_parse(generated_text)
    if result and isinstance(result, dict):
        # Проверяем, что все необходимые поля присутствуют
        missing_fields = set(structure['fields'].keys()) - set(result.keys())
        if missing_fields:
            # Добавляем недостающие поля с базовыми значениями
            for field in missing_fields:
                field_type = structure['fields'][field]['type']
                if field_type == 'string':
                    result[field] = f"Сгенерированное значение для {field}"
                elif field_type == 'integer':
                    result[field] = 0
                elif field_type == 'number':
                    result[field] = 0.0
                elif field_type == 'boolean':
                    result[field] = True
                elif field_type == 'array':
                    result[field] = []
                else:
                    result[field] = "значение"
        
        # КРИТИЧЕСКИ ВАЖНО: применяем строгую валидацию структуры JSON
        result = validate_and_fix_json_structure(result, structure)
        if not result:
            return None
        
        # Конвертируем типы данных для существующих полей
        result = convert_field_types(result, structure['fields'])
        
        return result
    
    # Если не удалось распарсить, возвращаем None для повторной попытки
    return None
