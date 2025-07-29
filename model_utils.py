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
    STRUCTURE_GENERATION_PROMPT, 
    DATA_GENERATION_PROMPT, 
    CONTEXTUAL_DATA_GENERATION_PROMPT,
    DATASET_TYPE_PROMPTS,
    JSON_REPAIR_PROMPT
)

def extract_json(text: str) -> str:
    """
    Простое извлечение JSON из текста.
    """
    # Убираем markdown
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Ищем первую { и последнюю }
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start == -1 or end <= start:
        return None
        
    json_text = text[start:end]
    
    # Убираем лишние запятые
    json_text = re.sub(r',\s*}', '}', json_text)
    json_text = re.sub(r',\s*]', ']', json_text)
    
    return json_text


def safe_json_parse(text: str) -> dict:
    """
    Безопасный парсинг JSON с несколькими попытками очистки.
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
    device: str = 'auto',          # 'auto' = выберет GPU, если есть
    use_8bit: bool = True,         # включить 8‑битное квантование
    token: str = None
):
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
    max_length: int = 1024  # Увеличено для более длинных промптов
) -> dict:
    """
    Создает простую структуру на основе описания датасета.
    """
    structure_prompt = STRUCTURE_GENERATION_PROMPT.format(description=description)

    with torch.inference_mode():
        inputs = tokenizer(
            structure_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,  # Увеличено для сложных структур
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Убираем исходный промпт
    if structure_prompt in generated_text:
        generated_text = generated_text.replace(structure_prompt, "").strip()
    
    # Извлекаем JSON
    json_text = extract_json(generated_text)
    if json_text:
        try:
            # Используем безопасный парсер
            structure = safe_json_parse(json_text)
            if isinstance(structure, dict) and 'fields' in structure and 'example_record' in structure:
                print(f"Модель создала структуру с полями: {list(structure['fields'].keys())}")
                return structure
        except json.JSONDecodeError:
            pass # Ошибка будет обработана ниже
    
    print("Не удалось получить структуру от модели. Пожалуйста, попробуйте переформулировать описание.")
    return None


def generate_chunk(
    tokenizer,
    model,
    structure,
    description: str = "",
    max_new_tokens: int = 200
) -> dict:
    """
    Генерирует один чанк данных на основе структуры.
    Принимает структуру как dict; возвращает словарь с данными.
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


