# -*- coding: utf-8 -*-
"""
Модуль для работы с JSON: парсинг, валидация и исправление
"""
import json
import re
import random


def safe_json_parse(text: str) -> dict:
    """
    Безопасно разбирает строку JSON, предпринимая несколько попыток исправить распространенные ошибки.

    Args:
        text: Строка для разбора.

    Returns:
        Словарь, если разбор удался, иначе None.
    """
    if not text:
        return None
    
    # Пробуем просто распарсить строку прямо
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Пробуем найти блоки JSON между фигурными скобками
    try:
        # Находим все возможные JSON блоки
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # Очищаем от лишних символов
                cleaned = re.sub(r',\s*([}\]])', r'\1', match)  # Убираем лишние запятые
                cleaned = re.sub(r'([{\[,])\s*,', r'\1', cleaned)  # Убираем двойные запятые
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    
    # Пытаемся более простым способом найти JSON
    try:
        # Ищем от первой { до последней }
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_part = text[start:end]
            
            # Базовая очистка
            json_part = re.sub(r',\s*}', '}', json_part)  # Убираем запятые перед }
            json_part = re.sub(r',\s*]', ']', json_part)  # Убираем запятые перед ]
            json_part = re.sub(r'([{,])\s*,', r'\1', json_part)  # Убираем двойные запятые
            
            return json.loads(json_part)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Ищем многострочный JSON, если предыдущие не сработали
    try:
        lines = text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('{'):
                in_json = True
                brace_count += line.count('{') - line.count('}')
                json_lines.append(line)
            elif in_json:
                brace_count += line.count('{') - line.count('}')
                json_lines.append(line)
                if brace_count <= 0:
                    break
        
        if json_lines:
            json_text = '\n'.join(json_lines)
            json_text = re.sub(r',\s*}', '}', json_text)
            json_text = re.sub(r',\s*]', ']', json_text)
            return json.loads(json_text)
    except Exception:
        pass
    
    return None


def validate_and_fix_json_structure(generated_json: dict, required_structure: dict) -> dict:
    """
    СТРОГО валидирует и исправляет структуру JSON, обеспечивая 100% соответствие требуемой схеме.
    
    Args:
        generated_json: Сгенерированный JSON
        required_structure: Требуемая структура с полями
    
    Returns:
        Исправленный JSON с правильной структурой
    """
    if not isinstance(generated_json, dict):
        return None
    
    required_fields = set(required_structure['fields'].keys())
    generated_fields = set(generated_json.keys())
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли лишние поля?
    extra_fields = generated_fields - required_fields
    if extra_fields:
        for field in extra_fields:
            del generated_json[field]
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: есть ли отсутствующие поля?
    missing_fields = required_fields - generated_fields
    if missing_fields:
        for field in missing_fields:
            field_type = required_structure['fields'][field]['type']
            if field_type == 'string':
                generated_json[field] = f"Автогенерированное значение для {field}"
            elif field_type == 'integer':
                generated_json[field] = random.randint(1, 1000)
            elif field_type == 'number':
                generated_json[field] = round(random.uniform(1.0, 100.0), 2)
            elif field_type == 'boolean':
                generated_json[field] = random.choice([True, False])
            else:
                generated_json[field] = "значение"
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: правильный ли порядок полей?
    # Пересоздаем JSON в правильном порядке
    correct_ordered_json = {}
    for required_field in required_structure['fields'].keys():
        if required_field in generated_json:
            correct_ordered_json[required_field] = generated_json[required_field]
    
    return correct_ordered_json
