# -*- coding: utf-8 -*-
"""
Конфигурация доступных моделей для генерации данных
"""

# Список доступных моделей
MODEL_CHOICES = [
    ("Qwen3-4B", "Qwen/Qwen3-4B"),
    ("Qwen3-8B", "Qwen/Qwen3-8B")
]

# Настройки генерации по умолчанию
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": False,
    "top_p": 0.95
}

# Настройки для генерации данных (более креативные)
DATA_GENERATION_CONFIG = {
    "temperature": 0.9,
    "top_p": 0.95,
    "do_sample": True
}
