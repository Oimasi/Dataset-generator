# -*- coding: utf-8 -*-
"""
Модуль для анализа полей данных: определение типов и конвертация значений
"""
import re
import random


def detect_field_type(field_name: str, example_value: str, original_description: str) -> str:
    """
    Определяет тип поля на основе его названия, примера и контекста.
    
    Args:
        field_name: Русское название поля
        example_value: Пример значения поля
        original_description: Исходное описание датасета от пользователя
    
    Returns:
        Тип поля: 'integer', 'number', 'string', 'boolean', 'array'
    """
    field_lower = field_name.lower()
    example_lower = example_value.lower() if example_value else ""
    description_lower = original_description.lower()
    
    # ПЕРВЫЙ ПРИОРИТЕТ: Проверяем на явно текстовые поля по примерам
    # Если в примере есть буквы (не цифры), запятые между словами, или текстовые значения - это строка
    if example_value and example_value.strip():
        # Убираем пробелы и проверяем содержимое
        clean_example = example_value.strip()
        
        # Если пример содержит буквы и не является чистым числом
        if re.search(r'[а-яА-ЯёЁa-zA-Z]', clean_example):
            # Исключение: если это явно диапазон чисел ("от 1 до 10", "1-5")
            if not (re.search(r'от\s+\d+.*до\s+\d+', example_lower) or 
                   re.search(r'^\d+\s*-\s*\d+$', clean_example)):
                return 'string'
        
        # Если пример содержит запятые между словами ("продукты, массаж, картошка")
        if ',' in clean_example and re.search(r'[а-яА-ЯёЁa-zA-Z]', clean_example):
            return 'string'
        
        # Если пример является чисто числовым (только цифры)
        if re.match(r'^\d+$', clean_example):
            return 'integer'
        
        # Если пример является числом с плавающей точкой
        if re.match(r'^\d+\.\d+$', clean_example):
            return 'number'
    
    # ВТОРОЙ ПРИОРИТЕТ: Определение по ключевым словам в названии поля
    # Явно текстовые поля
    string_keywords = ['название', 'наименование', 'имя', 'описание', 'комментарий', 'текст', 'адрес', 'email', 'телефон', 'name', 'title', 'description', 'comment', 'text', 'address']
    if any(keyword in field_lower for keyword in string_keywords):
        return 'string'
    
    # Числовые поля (только если нет явных текстовых признаков)
    integer_keywords = ['сумма', 'количество', 'число', 'цена', 'стоимость', 'возраст', 'год', 'рублей', 'руб', 'долларов', 'age', 'count', 'amount', 'price', 'cost', 'year']
    number_keywords = ['рейтинг', 'процент', 'rating', 'percent', 'score']
    boolean_keywords = ['статус', 'успешно', 'неуспешно', 'тип', 'является', 'status', 'success', 'failed']
    
    # Проверяем на числовые диапазоны в примере
    if re.search(r'от\s+\d+.*до\s+\d+', example_lower) or re.search(r'^\d+\s*-\s*\d+$', example_value.strip() if example_value else ''):
        return 'integer'  # Диапазон чисел -> integer
    
    # Проверяем на двоичные значения (0/1, true/false)
    if re.search(r'[01]\s*или\s*[01]', example_lower) or 'товар - 0' in example_lower or 'услуга - 1' in example_lower:
        return 'integer'  # Для совместимости с числовыми кодами
    
    # Проверяем ключевые слова в названии поля (только если пример не противоречит)
    if any(keyword in field_lower for keyword in integer_keywords):
        # Дополнительная проверка: если есть пример с текстом, игнорируем ключевое слово
        if not (example_value and re.search(r'[а-яА-ЯёЁa-zA-Z]', example_value)):
            return 'integer'
    
    if any(keyword in field_lower for keyword in number_keywords):
        if not (example_value and re.search(r'[а-яА-ЯёЁa-zA-Z]', example_value)):
            return 'number'
    
    if any(keyword in field_lower for keyword in boolean_keywords) and ('0' in example_lower or '1' in example_lower):
        return 'integer'
    
    # Проверяем на даты
    if re.search(r'дата|date', field_lower) or re.search(r'\d{4}-\d{2}-\d{2}', example_value if example_value else ''):
        return 'string'  # Даты как строки
    
    # По умолчанию - строка (безопасный выбор)
    return 'string'


def convert_field_types(data: dict, field_definitions: dict) -> dict:
    """
    Конвертирует значения полей к правильным типам данных.
    
    Args:
        data: Словарь с данными
        field_definitions: Определения полей с типами
    
    Returns:
        Обновленный словарь с правильными типами
    """
    converted_data = {}
    
    for field_name, value in data.items():
        if field_name not in field_definitions:
            converted_data[field_name] = value
            continue
        
        field_type = field_definitions[field_name]['type']
        
        try:
            if field_type == 'integer':
                # Пробуем конвертировать в целое число
                if isinstance(value, str):
                    # Ищем числа в строке
                    numbers = re.findall(r'\d+', value)
                    if numbers:
                        converted_data[field_name] = int(numbers[0])
                    else:
                        # Если не нашли число, генерируем случайное
                        converted_data[field_name] = random.randint(10, 100000)
                elif isinstance(value, (int, float)):
                    converted_data[field_name] = int(value)
                else:
                    converted_data[field_name] = random.randint(10, 100000)
                    
            elif field_type == 'number':
                # Конвертируем в число с плавающей точкой
                if isinstance(value, str):
                    # Ищем числа в строке
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        converted_data[field_name] = float(numbers[0])
                    else:
                        converted_data[field_name] = round(random.uniform(1.0, 10.0), 2)
                elif isinstance(value, (int, float)):
                    converted_data[field_name] = float(value)
                else:
                    converted_data[field_name] = round(random.uniform(1.0, 10.0), 2)
                    
            elif field_type == 'boolean':
                # Конвертируем в булево значение
                if isinstance(value, str):
                    value_lower = value.lower()
                    if value_lower in ['true', 'yes', '1', 'да']:
                        converted_data[field_name] = True
                    elif value_lower in ['false', 'no', '0', 'нет']:
                        converted_data[field_name] = False
                    else:
                        converted_data[field_name] = random.choice([True, False])
                elif isinstance(value, (int, float)):
                    converted_data[field_name] = bool(value)
                else:
                    converted_data[field_name] = bool(value)
                    
            elif field_type == 'string':
                # Преобразуем в строку
                converted_data[field_name] = str(value)
                
            else:
                # По умолчанию оставляем как есть
                converted_data[field_name] = value
                
        except (ValueError, TypeError):
            # Если конвертация не удалась, оставляем оригинальное значение
            converted_data[field_name] = value
    
    return converted_data


def enhance_field_description(field_name: str, example_value: str, original_description: str, field_type: str) -> str:
    """
    Обогащает описание поля на основе контекста из исходного описания датасета.
    
    Args:
        field_name: Русское название поля
        example_value: Пример значения поля
        original_description: Исходное описание датасета от пользователя
        field_type: Тип поля
    
    Returns:
        Обогащенное описание поля
    """
    # Ищем дополнительный контекст в исходном описании
    description_lower = original_description.lower()
    field_lower = field_name.lower()
    
    # Паттерны для поиска дополнительной информации о полях
    patterns = [
        # Ищем объяснения в скобках типа "товар - 0, услуга - 1"
        rf'{re.escape(field_lower)}[^\(]*\(([^\)]*-[^\)]*)\)',
        # Ищем объяснения после поля
        rf'{re.escape(field_lower)}[^,]*\(([^\)]+)\)',
        # Общий поиск дополнительной информации
        rf'{re.escape(field_lower)}[^,]*([^,]*)',
    ]
    
    additional_info = ""
    for pattern in patterns:
        match = re.search(pattern, description_lower)
        if match:
            info = match.group(1).strip()
            # Проверяем, содержит ли найденная информация полезные детали
            if any(char in info for char in ['-', ':', 'или', 'и']):
                additional_info = info
                break
    
    # Формируем итоговое описание с учетом типа
    base_description = f"{field_name.capitalize()}"
    
    if field_type == 'integer':
        base_description += " (целое число)"
    elif field_type == 'number':
        base_description += " (число)"
    elif field_type == 'boolean':
        base_description += " (логическое значение)"
    
    if additional_info and additional_info not in example_value:
        # Добавляем дополнительную информацию, если она не дублируется в примере
        base_description += f" ({additional_info})"
    
    if example_value and not example_value.startswith('Example for'):
        base_description += f". Пример: {example_value}"
    
    return base_description
