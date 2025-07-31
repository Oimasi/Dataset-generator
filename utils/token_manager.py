# -*- coding: utf-8 -*-
"""
Модуль для управления токенами Hugging Face
"""
import os
import base64

ENV_FILE = ".env"
TOKEN_KEY = "HF_TOKEN"


def load_env():
    """Загружает переменные из .env файла"""
    env_vars = {}
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except Exception:
            pass
    return env_vars


def save_env(key, value):
    """Сохраняет переменную в .env файл с простым кодированием"""
    try:
        # Кодируем токен в base64 чтобы не светился в открытом виде
        encoded_value = base64.b64encode(value.encode()).decode()
        
        env_vars = load_env()
        env_vars[key] = encoded_value
        
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.write("# Файл переменных окружения\n")
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        return True
    except Exception:
        return False


def load_token():
    """Загружает токен из .env файла"""
    env_vars = load_env()
    if TOKEN_KEY in env_vars:
        try:
            # Декодируем токен обратно
            return base64.b64decode(env_vars[TOKEN_KEY]).decode()
        except Exception:
            return None
    return None


def save_token(token):
    """Сохраняет токен в .env файл."""
    return save_env(TOKEN_KEY, token)


def get_token():
    """Получает токен: сначала пытается загрузить из файла, если не удается, запрашивает у пользователя."""
    token = load_token()
    if token:
        # Автоматически используем сохраненный токен без запроса
        return token
    
    # Запрашиваем новый токен только если его нет
    token = input("Введите ваш токен Hugging Face (или оставьте пустым): ").strip()
    if token:
        save_token(token)
        print("Токен сохранен для будущих запусков.")
    return token
