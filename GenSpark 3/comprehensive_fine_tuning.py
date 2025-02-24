import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

#Check if CUDA is available
print(torch.cuda.is_available()) # Should return True if a GPU is available

# Load the model and tokenizer
model_name = 'distilbert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#prepare the data
dataset = load_dataset('imdb')
def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

#set the training arguments
training_args = TrainingArguments(
    output_dir='./results', evaluation_strategy='epoch', num_train_epochs=3, 
    learning_rate=2e-5, per_device_train_batch_size=16, weight_decay=0.015
)

#setup the trainer
trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['test']
)

#train the model
trainer.train()

#save the model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

#evaluate the model
results = trainer.evaluate()
print(results)

#sklearn classification report
predictions = trainer.predict(tokenized_datasets["test"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_datasets["test"]["label"]
print(classification_report(y_true, y_pred))