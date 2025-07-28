
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
def load_model(model_id, device="cpu", token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token)
    model.to(device)
    return tokenizer, model

def structure_prompt(tokenizer, model, description: str, max_length: int = 128, device: str = 'cpu') -> str:
    """
    Формирует четкий промпт для генерации JSON-структуры датасета.

    Args:
        tokenizer: Токенизатор модели.
        model: Модель для структурирования.
        description: Пользовательское описание датасета.
        max_length: Максимальная длина генерируемого текста.
        device: Устройство для вычислений.

    Returns:
        сгенерированный текст-промпт
    """
    system_instruction = (
        "Ты помогаешь формировать четкий JSON-шаблон для генерации датасета на основе описания."
    )
    input_text = system_instruction + "\nОписание: " + description + "\nШаблон:"
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    prompt_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Оставляем после 'Шаблон:'
    return prompt_text.split('Шаблон:')[-1].strip()

def generate_chunk(tokenizer, model, prompt: str, max_length: int = 256, device: str = 'cpu') -> dict:
    """
    Генерирует один JSON-чанк данных по заданному промпту.

    Args:
        tokenizer: Токенизатор модели.
        model: Модель для генерации.
        prompt: Промпт, описывающий формат и содержание.
        max_length: Максимальная длина ответа.
        device: Устройство для вычислений.

    Returns:
        Python-словарь, полученный из сгенерированного JSON.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Пытаемся вырезать JSON из текста
    try:
        json_start = text.find('{')
        json_text = text[json_start:]
        return json.loads(json_text)
    except Exception:
        # Возвращаем необработанный текст в случае ошибки
        return {'__raw__': text}