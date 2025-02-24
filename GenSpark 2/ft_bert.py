import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import transformers

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Confirm whether GPU is used

# Load the model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess the dataset
dataset = load_dataset("imdb")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments with progress bar settings
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
)

# Force transformers to show a progress bar
transformers.utils.logging.set_verbosity_info()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

results = trainer.evaluate()
print("Evaluation results:", results)

# Generate predictions and print a classification report
predictions = trainer.predict(tokenized_datasets["test"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_datasets["test"]["label"]
print("Classification Report:\n", classification_report(y_true, y_pred))
