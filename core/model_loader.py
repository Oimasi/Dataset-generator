# -*- coding: utf-8 -*-
"""
Модуль для загрузки и настройки языковых моделей Hugging Face
"""
import warnings
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Выключаем все предупреждения, чтобы не мешали
warnings.filterwarnings('ignore')

# Убираем предупреждения TensorFlow, чтобы экран не засоряли
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


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
