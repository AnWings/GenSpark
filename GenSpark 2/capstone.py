import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

# Load the dataset
train_df = pd.read_csv(r"C:\Python Codes\GenSpark 2\train.csv", encoding="ISO-8859-1")
test_df = pd.read_csv(r"C:\Python Codes\GenSpark 2\test.csv", encoding="ISO-8859-1")

# Data Preprocessing and cleaning
test_df['selected_text'] = None
merged_df = pd.concat([train_df, test_df], ignore_index=True)

merged_df.columns = merged_df.columns.str.strip()
merged_df = merged_df[['text', 'sentiment']]

# Drop rows with missing text or sentiment values
merged_df = merged_df.dropna(subset=['text', 'sentiment'])

# Convert sentiment labels to integers (0 for negative, 1 for positive)
label_map = {'negative': 0, 'positive': 1}
merged_df['sentiment'] = merged_df['sentiment'].map(label_map)

# Drop any rows where mapping resulted in None and ensure integer type
merged_df = merged_df.dropna(subset=['sentiment'])
merged_df['sentiment'] = merged_df['sentiment'].astype(int)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(merged_df)

# Split dataset into train and validation sets
dataset = dataset.train_test_split(test_size=0.2)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess function for tokenization and adding labels
def preprocess_function(examples):
    encoding = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )
    if 'sentiment' in examples:
        encoding["labels"] = examples["sentiment"]
    else:
        raise ValueError("The 'sentiment' column is missing in the dataset.")
    return encoding

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Debug: Print tokenized dataset columns to verify 'labels' exist
print("Tokenized dataset columns:", tokenized_datasets["train"].column_names)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model.to(device),  # Move model to the correct device (GPU/CPU)
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Sklearn classification report
predictions = trainer.predict(tokenized_datasets["test"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_datasets["test"]["sentiment"]
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')