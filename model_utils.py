import warnings
import os
import re

# Выключаем все предупреждения, чтобы не мешали
warnings.filterwarnings('ignore')

# Убираем предупреждения TensorFlow, чтобы экран не засоряли
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import torch
from prompts import (
    EXTRACT_FIELDS_PROMPT,
    DATA_GENERATION_PROMPT,
    CONTEXTUAL_DATA_GENERATION_PROMPT,
    DATASET_TYPE_PROMPTS,
    JSON_REPAIR_PROMPT
)



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
                        import random
                        converted_data[field_name] = random.randint(10, 100000)
                elif isinstance(value, (int, float)):
                    converted_data[field_name] = int(value)
                else:
                    import random
                    converted_data[field_name] = random.randint(10, 100000)
                    
            elif field_type == 'number':
                # Конвертируем в число с плавающей точкой
                if isinstance(value, str):
                    # Ищем числа в строке
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        converted_data[field_name] = float(numbers[0])
                    else:
                        import random
                        converted_data[field_name] = round(random.uniform(1.0, 10.0), 2)
                elif isinstance(value, (int, float)):
                    converted_data[field_name] = float(value)
                else:
                    import random
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
                        import random
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


def load_model(
    model_id: str,
    device: str = 'auto',
    use_8bit: bool = True,
    token: str = None
):
    """
    Загружает модель и токенизатор Hugging Face с возможностью квантования.

    Args:
        model_id (str): Идентификатор модели на Hugging Face Hub.
        device (str): Устройство для загрузки модели ('auto', 'cuda', 'cpu').
        use_8bit (bool): Использовать ли 8-битное квантование для уменьшения потребления памяти.
        token (str): Токен Hugging Face для доступа к закрытым моделям.

    Returns:
        Кортеж (tokenizer, model).
    """
    # Теперь используем token вместо use_auth_token
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    
    # Создаем конфигурацию квантования
    quantization_config = None
    if use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    model.eval()
    return tokenizer, model


def structure_prompt_v2(
    tokenizer,
    model,
    description: str,
    max_length: int = 4_096
) -> dict:
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
        Словарь со структурой датасета или None в случае неудачи.
    """
    # Шаг 1: используем модель для извлечения пар "русское_название:english_key"
    prompt = EXTRACT_FIELDS_PROMPT.format(description=description)
    with torch.inference_mode():
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and clean the model's response
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "")
    
    field_pairs_str = generated_text.split("Результат:")[-1].strip()
    
    if not field_pairs_str or ':' not in field_pairs_str:
        return None

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
            example = re.sub(r'["\\n\\r]+.*$', '', example).strip()

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
            return None

        structure = {
            'fields': fields,
            'example_record': example_record
        }
        
        return structure

    except Exception:
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
                import random
                generated_json[field] = random.randint(1, 1000)
            elif field_type == 'number':
                import random
                generated_json[field] = round(random.uniform(1.0, 100.0), 2)
            elif field_type == 'boolean':
                import random
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


def generate_chunk(
    tokenizer,
    model,
    structure,
    description: str = "",
    max_new_tokens: int = 1024
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
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,  # Больше креативности для данных
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Убираем исходный промпт
    if generation_prompt in generated_text:
        generated_text = generated_text.replace(generation_prompt, "").strip()
    
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


