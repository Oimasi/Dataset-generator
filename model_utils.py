
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

def load_model(
    model_id: str,
    device: str = 'auto',          # 'auto' = выберет GPU, если есть
    use_8bit: bool = True,         # включить 8‑битное квантование
    token: str = None
):
    # 1. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)

    # 2. Модель с оптимальными параметрами для инференса
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=token,
        device_map="auto",               # автоматически распределит по доступным устройствам
        low_cpu_mem_usage=True,          # снизит пиковое потребление CPU-памяти при загрузке
        torch_dtype=torch.float16,       # загрузит веса в FP16
        load_in_8bit=use_8bit            # при наличии bitsandbytes — квантование до 8‑бит
    )

    model.eval()
    return tokenizer, model

def structure_prompt_v2(
    tokenizer, 
    model, 
    description: str, 
    max_length: int = 1024, 
    device: str = 'cpu'
) -> str:
    instruction = (
        "Ты — помощник по созданию структуры синтетических датасетов. "
        "На основе пользовательского описания ты должен выдать чёткий и полный JSON-шаблон с примерами значений, "
        "где для каждого поля указывается тип данных и пример. "
        "Не добавляй лишних пояснений — только чистый JSON-шаблон.\n\n"
        "Пример:\n"
        "Описание: Датасет о фильмах: название, жанр, год выпуска, рейтинг.\n"
        "Шаблон:\n"
        "{\n"
        "  \"title\": \"Inception\",\n"
        "  \"genre\": \"Sci-Fi\",\n"
        "  \"year\": 2010,\n"
        "  \"rating\": 8.8\n"
        "}\n\n"
        f"Описание: {description}\n"
        "Шаблон:\n"
    )
    # Переносим тензоры на нужное устройство
    inputs = tokenizer(instruction, return_tensors='pt', truncation=True).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Обрезаем префикс
    json_part = result.split('Шаблон:')[-1].strip()
    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        return {"raw": json_part}


def generate_chunk(tokenizer, model, prompt: str, max_length: int = 256) -> dict:
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()
    try:
        return json.loads(generated)
    except json.JSONDecodeError:
        return {"raw": generated}