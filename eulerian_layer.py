import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

# =====================================================================
# 1. НАСТРОЙКА УСТРОЙСТВА (GPU)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# =====================================================================
# 2. ГОМЕОСТАТИЧЕСКИЙ СЛОЙ С ЭНДОГЕННОЙ ТЕМПЕРАТУРОЙ
# =====================================================================
class HomeostaticLayer(nn.Module):
    def __init__(self, embed_dim, critical_temp=5.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.critical_temp = critical_temp

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_gate = nn.Linear(embed_dim, 1)

    def forward(self, x, prev_temp, prev_amnesia):
        # Аналог внимания: насколько текущий токен "резонирует" с контекстом
        q = self.query_proj(x[:, -1, :])
        k = self.key_proj(x).mean(dim=1)
        info_density = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))

        # Изменение температуры зависит от насыщенности контекста
        delta_temp = torch.tanh(self.temp_gate(x[:, -1, :]))
        new_temp = prev_temp + delta_temp * info_density
        new_temp = torch.clamp(new_temp, min=0.2, max=6.0)

        # Ворота амнезии: накопительный, обратимый фильтр
        amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
        new_amnesia = prev_amnesia * (1 - amnesia_gate) + amnesia_gate

        # Температура модулирует отклик слоя
        x_modified = x * torch.exp(-new_temp.unsqueeze(1) / 10.0)

        return x_modified, new_temp, new_amnesia


class HomeostaticTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([HomeostaticLayer(embed_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.num_layers = num_layers

    def forward(self, x, temps=None, amnesias=None):
        batch_size = x.size(0)
        if temps is None:
            temps = [torch.ones(batch_size, 1, device=device) * 1.0 for _ in range(self.num_layers)]
        if amnesias is None:
            amnesias = [torch.zeros(batch_size, 1, device=device) for _ in range(self.num_layers)]

        out = self.embedding(x)
        new_temps, new_amnesias = [], []

        for i, layer in enumerate(self.layers):
            out, t, a = layer(out, temps[i], amnesias[i])
            new_temps.append(t)
            new_amnesias.append(a)

        logits = self.fc_out(out)
        return logits, new_temps, new_amnesias


# =====================================================================
# 3. РАСШИРЕННЫЙ СЛОВАРЬ (5000 ТОКЕНОВ)
# =====================================================================
print("Загрузка TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:50000]")

VOCAB_SIZE = 5000
all_text = " ".join(dataset["text"][:5000])
words = all_text.lower().split()
vocab_counts = Counter(words)
vocab = ["<pad>", "<unk>"] + [word for word, _ in vocab_counts.most_common(VOCAB_SIZE - 2)]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

def tokenize(text):
    return [word2idx.get(w, word2idx["<unk>"]) for w in text.lower().split()][:32]

print(f"Словарь расширен до: {len(vocab)} токенов")

tokenized_data = [tokenize(t) for t in dataset["text"][:15000] if len(tokenize(t)) > 5]
max_len = max(len(t) for t in tokenized_data)
padded_data = [t + [word2idx["<pad>"]] * (max_len - len(t)) for t in tokenized_data]

X_train = torch.tensor([t[:-1] for t in padded_data], dtype=torch.long).to(device)
Y_train = torch.tensor([t[1:] for t in padded_data], dtype=torch.long).to(device)

# =====================================================================
# 4. ТРЕНИРОВКА (25 ДНЕЙ)
# =====================================================================
model = HomeostaticTransformer(vocab_size=len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

epochs = 25
batch_size = 256
print(f"Начало обучения ({epochs} дней)...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    permutation = torch.randperm(X_train.size(0))

    last_temps = []
    total_amnesias = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], Y_train[indices]

        optimizer.zero_grad()
        logits, temps, amnesias = model(batch_x)

        loss = criterion(logits.view(-1, len(vocab)), batch_y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        last_temps = [t.mean().item() for t in temps]
        total_amnesias = sum(a.mean().item() for a in amnesias)

    avg_loss = total_loss / (X_train.size(0) / batch_size)
    print(f"День {epoch+1}: LM loss={avg_loss:.3f}, темп-ра верхнего слоя={last_temps[-1]:.2f}, амнезий={int(total_amnesias)}")

# =====================================================================
# 5. УМНАЯ ТВОРЧЕСКАЯ ГЕНЕРАЦИЯ (С СЕМПЛИРОВАНИЕМ)
# =====================================================================
print("\nГенерация с телеметрией...")
model.eval()

start_seq = [word2idx.get(w, word2idx["<unk>"]) for w in ["once", "upon", "a", "time"]]
input_tensor = torch.tensor([start_seq], dtype=torch.long).to(device)

history_temps = [[] for _ in range(model.num_layers)]
history_amnesia = [[] for _ in range(model.num_layers)]

generated_words = ["once", "upon", "a", "time"]

t_state = [torch.ones(1, 1, device=device) * 1.2 for _ in range(model.num_layers)]
a_state = [torch.zeros(1, 1, device=device) for _ in range(model.num_layers)]

with torch.no_grad():
    for _ in range(60):
        logits, t_state, a_state = model(input_tensor, t_state, a_state)

        for layer_idx in range(model.num_layers):
            history_temps[layer_idx].append(t_state[layer_idx].mean().item())
            history_amnesia[layer_idx].append(a_state[layer_idx].mean().item())

        next_token_logits = logits[0, -1, :]
        next_token_logits[word2idx["<unk>"]] -= 10.0
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated_words.append(idx2word.get(next_token, "<unk>"))
        next_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

print("\n--- Сгенерированный текст ---")
print(" ".join(generated_words))
print("-----------------------------\n")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for i in range(model.num_layers):
    plt.plot(history_temps[i], label=f"Слой {i}")
plt.axhline(y=1.5, color='gray', linestyle='--')
plt.axhline(y=5.0, color='red', linestyle='--', label="Критическая")
plt.title("Динамика температуры при генерации")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(model.num_layers):
    plt.plot(history_amnesia[i], label=f"Слой {i}")
plt.title("Ворота амнезии")
plt.legend()
plt.show()
