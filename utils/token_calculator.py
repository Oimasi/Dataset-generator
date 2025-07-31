# -*- coding: utf-8 -*-
"""
Модуль для расчета оптимального количества токенов для генерации
"""
import re


def calculate_max_tokens(description: str, base_tokens: int = 256, min_tokens: int = 128, max_tokens: int = 1024) -> int:
    """
    Динамически вычисляет оптимальное количество новых токенов на основе сложности описания.
    
    Улучшенная логика расчета:
    1. Подсчет полей (основной фактор)
    2. Длина описания (дополнительный фактор)
    3. Сложность примеров и пояснений
    4. Специальные конструкции
    
    Args:
        description: Описание датасета
        base_tokens: Базовое количество токенов (по умолчанию 512)
        min_tokens: Минимальное количество токенов (по умолчанию 256)
        max_tokens: Максимальное количество токенов (по умолчанию 2048)
    
    Returns:
        Оптимальное количество токенов для генерации
    """
    if not description or not description.strip():
        return base_tokens
    
    desc_length = len(description.strip())
    
    # 1. ОСНОВНОЙ ФАКТОР: Количество полей
    # Считаем запятые как разделители полей
    field_count = description.count(',') + 1  # +1 потому что первое поле без запятой
    
    # Базовый расчет: 40-80 токенов на поле
    if field_count <= 3:
        tokens_per_field = 60  # Простые поля
    elif field_count <= 6:
        tokens_per_field = 40  # Средняя сложность
    elif field_count <= 10:
        tokens_per_field = 30  # Много полей
    else:
        tokens_per_field = 20  # Очень много полей
    
    tokens = field_count * tokens_per_field
    
    # 2. ДОПОЛНИТЕЛЬНЫЙ ФАКТОР: Длина описания
    if desc_length > 300:
        tokens = int(tokens * 1.3)  # Длинные описания требуют больше токенов
    elif desc_length > 150:
        tokens = int(tokens * 1.15)  # Средние описания
    
    # 3. СЛОЖНОСТЬ: Примеры в скобках
    parentheses_pairs = min(description.count('('), description.count(')'))
    if parentheses_pairs > 0:
        # Каждая пара скобок добавляет 20-30 токенов
        tokens += parentheses_pairs * 25
    
    # 4. СЛОЖНОСТЬ: Ключевые слова, указывающие на сложные конструкции
    complex_patterns = [
        'например', 'пример', 'к примеру',
        'может быть', 'или', 'либо',
        'от', 'до', 'диапазон',
        'если', 'когда', 'где',
        'обязательно', 'необходимо'
    ]
    
    complexity_score = 0
    for pattern in complex_patterns:
        if pattern in description.lower():
            complexity_score += 1
    
    # Каждое сложное слово добавляет 15 токенов
    tokens += complexity_score * 15
    
    # 5. СПЕЦИАЛЬНЫЕ КОНСТРУКЦИИ: Диапазоны чисел ("от 1 до 10")
    number_ranges = len(re.findall(r'от\s+\d+.*?до\s+\d+', description.lower()))
    number_ranges += len(re.findall(r'\d+\s*-\s*\d+', description))
    tokens += number_ranges * 20
    
    # 6. СПЕЦИАЛЬНЫЕ КОНСТРУКЦИИ: Варианты выбора ("0 или 1", "да или нет")
    choice_patterns = len(re.findall(r'\d+\s+или\s+\d+', description.lower()))
    choice_patterns += len(re.findall(r'\b(да|нет)\s+или\s+(да|нет)\b', description.lower()))
    tokens += choice_patterns * 20
    
    # 7. МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ: Обеспечиваем достаточно токенов для JSON структуры
    # Базовый JSON требует минимум токенов
    json_overhead = 50 + field_count * 10  # Базовая структура + ключи
    tokens = max(tokens, json_overhead)
    
    # Ограничиваем результат в разумных пределах
    final_tokens = max(min_tokens, min(tokens, max_tokens))
    
    return final_tokens
