import warnings
import os
import re

# Отключаем все предупреждения
warnings.filterwarnings('ignore')

# Отключаем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    
    # Попытка 1: прямой парсинг
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Попытка 2: ищем JSON блок между фигурными скобками
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
    
    # Попытка 3: более простой поиск
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
    
    # Попытка 4: поиск многострочного JSON
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
    # Используем новый параметр token вместо use_auth_token
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
    max_length: int = 1024
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
    # Step 1: Use the model to extract "russian_name:english_key" pairs
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
            max_new_tokens=200,
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
        
        # Use regex to robustly find all "russian_name:english_key:example" triplets
        # Останавливаемся на запятой или конце строки
        pattern = re.compile(r'([а-яА-ЯёЁ -]+):([a-zA-Z_]+):([^,]*?)(?=,|$)')
        matches = pattern.findall(field_pairs_str)

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

            fields[eng_key] = {
                'type': 'string',
                'description': f'{rus_name.capitalize()}. Пример: {example_value}' if example else rus_name.capitalize(),
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


def generate_chunk(
    tokenizer,
    model,
    structure,
    description: str = "",
    max_new_tokens: int = 200
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
            example_record=example_record_str
        )
    else:
        generation_prompt = DATA_GENERATION_PROMPT.format(
            field_descriptions=chr(10).join(field_descriptions),
            example_record=example_record_str
        )

    with torch.inference_mode():
        inputs = tokenizer(
            generation_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
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
                elif field_type == 'number':
                    result[field] = 0
                elif field_type == 'boolean':
                    result[field] = True
                elif field_type == 'array':
                    result[field] = []
                else:
                    result[field] = "значение"
        
        print(f"Успешно создана запись с полями: {list(result.keys())}")
        return result
    
    # Если не удалось распарсить, возвращаем None для повторной попытки
    print("Не удалось распарсить сгенерированные данные, будет произведена повторная попытка...")
    return None


