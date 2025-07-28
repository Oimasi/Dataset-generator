import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Простая заглушка генераторной нейросети
class SimpleGenerator(nn.Module):
    def __init__(self, noise_dim: int, out_dim: int, hidden_dim: int = 128):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.model(z)


def train_generator(
    generator: nn.Module,
    num_steps: int,
    noise_dim: int,
    lr: float,
    device: torch.device,
    log_callback=None
):
    generator.train()
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        noise = torch.randn(1, noise_dim, device=device)
        fake = generator(noise)
        target = torch.zeros_like(fake)
        loss = criterion(fake, target)
        loss.backward()
        optimizer.step()
        if log_callback and (step % max(1, num_steps // 10) == 0 or step == 1):
            log_callback(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")


def generate_dataset(
    generator: nn.Module,
    noise_dim: int,
    num_samples: int,
    device: torch.device
) -> np.ndarray:
    generator.eval()
    zs = torch.randn(num_samples, noise_dim, device=device)
    with torch.no_grad():
        data = generator(zs).cpu().numpy()
    return data


def save_dataset(
    data: np.ndarray,
    output_dir: str,
    filename: str = "synthetic_data.npy"
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.save(path, data)
    return path


class App:
    def __init__(self, root):
        self.root = root
        root.title("Генератор синтетического датасета")
        self.create_widgets()

    def create_widgets(self):
        params = [
            ("Noise dim", "16"),
            ("Output dim", "8"),
            ("Num samples", "1000"),
            ("Train steps", "100"),
            ("Learning rate", "0.001"),
        ]
        self.entries = {}
        for i, (label, default) in enumerate(params):
            ttk.Label(self.root, text=label).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            entry = ttk.Entry(self.root)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label] = entry

        ttk.Button(self.root, text="Select Output Folder", command=self.select_folder).grid(row=len(params), column=0, pady=10)
        self.output_var = tk.StringVar(value=os.getcwd())
        ttk.Label(self.root, textvariable=self.output_var).grid(row=len(params), column=1, pady=10)

        self.log = tk.Text(self.root, height=10, width=50)
        self.log.grid(row=len(params)+1, column=0, columnspan=2, padx=5, pady=5)

        ttk.Button(self.root, text="Generate", command=self.run).grid(row=len(params)+2, column=0, columnspan=2, pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def log_callback(self, message: str):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.root.update()

    def run(self):
        try:
            noise_dim = int(self.entries["Noise dim"].get())
            out_dim = int(self.entries["Output dim"].get())
            num_samples = int(self.entries["Num samples"].get())
            train_steps = int(self.entries["Train steps"].get())
            lr = float(self.entries["Learning rate"].get())
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числа")
            return

        device = torch.device("cpu")
        gen = SimpleGenerator(noise_dim, out_dim).to(device)
        self.log.delete(1.0, tk.END)
        self.log_callback("Starting training...")
        train_generator(gen, train_steps, noise_dim, lr, device, log_callback=self.log_callback)
        self.log_callback("Training finished. Generating data...")
        data = generate_dataset(gen, noise_dim, num_samples, device)
        path = save_dataset(data, self.output_var.get())
        self.log_callback(f"Dataset saved to {path}")
        messagebox.showinfo("Готово", f"Датасет сохранён:\n{path}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
