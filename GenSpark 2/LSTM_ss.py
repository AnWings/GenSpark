import torch
import torch.nn as nn
import numpy as np
import urllib.request
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

if torch.cuda.is_available():
    print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

# Download Shakespeare text
url = "https://www.gutenberg.org/files/100/100-0.txt"
file_path = "shakespeare.txt"
urllib.request.urlretrieve(url, file_path)
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"^\s*.*?\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL)
text = re.sub(r"\*\*\* END OF.*?$", "", text, flags=re.DOTALL)

# Convert text to lowercase
text = text.lower()

#print(f"Text length: {len(text)}")

chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

encoded_text = np.array([char2idx[c] for c in text], dtype=np.int64)

#print(f"Vocabulary Size: {len(chars)}")
#print(f"Sample encoded text: {encoded_text[:50]}")

sequence_length = 50
batch_size = 64

class ShakespeareDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.text[idx:idx + self.sequence_length], dtype=torch.long)
        target = torch.tensor(self.text[idx + self.sequence_length], dtype=torch.long)
        return input_seq, target

# Create dataset
dataset = ShakespeareDataset(encoded_text, sequence_length)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#print(f"Total sequences: {len(dataset)}")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#print(f"Training sequences: {len(train_dataset)}")
#print(f"Validation sequences: {len(val_dataset)}")

# Define LSTM model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output[:, -1, :])  # Take only the last output
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

# Model parameters
vocab_size = len(chars)
embed_dim = 128
hidden_dim = 256
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(batch_size)
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader), leave=False)

    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = tuple(h.detach() for h in hidden)
        
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "shakespeare_lstm.pth")
print("Model saved successfully!")

def generate_text(model, seed_text, length=200, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([char2idx[c] for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated_text = seed_text

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            output = output.squeeze(0) / temperature  # Adjust temperature
            probabilities = F.softmax(output, dim=-1).cpu().numpy()
            predicted_idx = np.random.choice(len(chars), p=probabilities)
        
        generated_text += idx2char[predicted_idx]
        input_seq = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)

    return generated_text

# Example: Generate text with different temperatures
seed = "to be, or not to be"
print("Low Temperature (Predictable):")
print(generate_text(model, seed, temperature=0.5))

print("\nHigh Temperature (Creative):")
print(generate_text(model, seed, temperature=1.5))