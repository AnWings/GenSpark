import nltk
nltk.download('punkt')  # Ensure correct tokenizer setup
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel

# Sample text
text = """
Eternity Could Wait
The time had come to say goodbye, but Liam didn’t want to let go. Sixty-three years of marriage would do that to you. He looked down on his wife, her eyes closed, her hands across her chest. It almost brought tears to his eyes again. He reached out and brushed her cheek.
Her eyes fluttered open. She searched the room, but nope, her husband was still gone. She was absolutely sure she had felt him. She felt the tears well up again. When will they ever stop? She rolled over to cry silently into her pillow. Why did he have to leave her?
He saw her look straight through him, not seeing his ghostly spirit. He knew he should go on to whatever fate he had waiting, but well, sixty-three years of marriage. Instead, he blew her a kiss and contented himself with watching over her. Eternity could wait.
"""

# Load tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize text using BERT tokenizer
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Get BERT's tokenized words

# Extract token embeddings
embeddings = []
valid_tokens = []  # To keep track of tokens with embeddings

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0]  # Shape: (seq_len, hidden_dim)

    for i, token in enumerate(tokens):
        # Skip padding tokens
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            embeddings.append(hidden_states[i].numpy())
            valid_tokens.append(token)

# Convert embeddings to NumPy array
embeddings = np.array(embeddings)

# Ensure t-SNE perplexity does not exceed the number of points - 1
perplexity = min(30, len(embeddings) - 1)
tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot embeddings
plt.figure(figsize=(10, 8))
for i, token in enumerate(valid_tokens):  # Use valid tokens only
    x, y = reduced_embeddings[i]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x + 0.01, y + 0.01, token, fontsize=9)

plt.title('Token Embeddings Visualization (t-SNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
