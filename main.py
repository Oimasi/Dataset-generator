import os
import json
import tkinter as tk
import torch

from tkinter import ttk, filedialog, messagebox
from model_utils import load_model, structure_prompt, generate_chunk
MODEL_CHOICES = [
    ("GPT-2 Small (124M)", "gpt2"),
    ("DistilGPT-2 (82M)", "distilgpt2"),
    ("TinyLlama 1.1B Chat v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Mistral Small 3.1 (24B)", "mistralai/Mistral-7B-Instruct-v0.1"),
    ("Qwen3-4B", "Qwen/Qwen3-4B")
]

def assemble_dataset(chunks: list, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

class App:
    def __init__(self, root):
        self.root = root
        root.title("Генератор датасетов на нейросетях")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.create_widgets()

    def create_widgets(self):
        # Модель для структурирования промпта
        ttk.Label(self.root, text="Model for structuring prompt:")\
            .grid(row=0, column=0, sticky=tk.W)
        struct_names = [name for name, _ in MODEL_CHOICES]
        self.struct_model = ttk.Combobox(
            self.root, values=struct_names, state="readonly"
        )
        self.struct_model.current(0)
        self.struct_model.grid(row=0, column=1, sticky=tk.EW)

        # Модель для генерации данных
        ttk.Label(self.root, text="Model for data generation:")\
            .grid(row=1, column=0, sticky=tk.W)
        self.gen_model = ttk.Combobox(
            self.root, values=struct_names, state="readonly"
        )
        self.gen_model.current(1)
        self.gen_model.grid(row=1, column=1, sticky=tk.EW)

        # Токен Hugging Face 
        ttk.Label(self.root, text="Hugging Face Token:")\
            .grid(row=2, column=0, sticky=tk.W)
        self.token_entry = ttk.Entry(self.root, show="*")
        self.token_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)

        # Описание датасета
        ttk.Label(self.root, text="Dataset description:")\
            .grid(row=3, column=0, sticky=tk.NW)
        self.desc_text = tk.Text(self.root, height=5, width=40)
        self.desc_text.grid(row=3, column=1, padx=5, pady=5)

        # Количество чанков
        ttk.Label(self.root, text="Number of chunks:")\
            .grid(row=4, column=0, sticky=tk.W)
        self.chunk_entry = ttk.Entry(self.root)
        self.chunk_entry.insert(0, "10")
        self.chunk_entry.grid(row=4, column=1, sticky=tk.EW)

        # Папка для сохранения
        ttk.Button(self.root, text="Select output folder", command=self.select_folder)\
            .grid(row=5, column=0)
        self.output_var = tk.StringVar(value=os.getcwd())
        ttk.Label(self.root, textvariable=self.output_var)\
            .grid(row=5, column=1, sticky=tk.W)

        # Информация о девайсе
        self.device_label = ttk.Label(self.root, text=f"Device: {self.device.upper()}")
        self.device_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Кнопка запуска
        ttk.Button(self.root, text="Generate Dataset", command=self.run)\
            .grid(row=7, column=0, columnspan=2, pady=10)

        # Лог
        self.log_text = tk.Text(self.root, height=10, width=60)
        self.log_text.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

        for i in range(2):
            self.root.grid_columnconfigure(i, weight=1)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def run(self):
        token = self.token_entry.get().strip()
        if token:
            os.environ['HUGGINGFACE_TOKEN'] = token

        desc = self.desc_text.get(1.0, tk.END).strip()
        if not desc:
            messagebox.showerror("Error", "Please enter dataset description.")
            return
        try:
            n_chunks = int(self.chunk_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of chunks.")
            return

        struct_id = dict(MODEL_CHOICES)[self.struct_model.get()]
        gen_id    = dict(MODEL_CHOICES)[self.gen_model.get()]

        self.log(f"Using device: {self.device.upper()}")
        self.log(f"Loading structure model: {struct_id}")
        tok_struct, model_struct = load_model(struct_id, device=self.device, token=token)
        self.log(f"Loading generation model: {gen_id}")
        tok_gen, model_gen       = load_model(gen_id, device=self.device, token=token)

        self.log("Structuring prompt...")
        prompt = structure_prompt(tok_struct, model_struct, desc)
        self.log(f"Structured prompt: {prompt}")

        dataset = []
        for i in range(n_chunks):
            self.log(f"Generating chunk {i+1}/{n_chunks}...")
            chunk = generate_chunk(tok_gen, model_gen, prompt)
            dataset.append(chunk)

        output_path = os.path.join(self.output_var.get(), "synthetic_dataset.json")
        assemble_dataset(dataset, output_path)
        self.log(f"Dataset saved to {output_path}")
        messagebox.showinfo("Done", f"Dataset saved:\n{output_path}")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()