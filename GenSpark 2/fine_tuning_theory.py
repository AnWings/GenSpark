# Transfer learning can be explained through the analogy of a Michelin-star chef.
# Imagine a chef who, after mastering Western cuisine, now endeavors to make sushi.
# The chef’s prior experience is valuable in the transfer of knowledge: from handling knives
# to pairing flavors in a visually pleasing way. However, sushi-making comes with its own set
# of challenges: raw fish handling, neat vinegared rice, rolling, and more. While the chef will
# need to learn these new skills, many of the fundamentals, such as precision, attention to detail,
# and value, will carry over into the new challenge.
# 
# In machine learning, this is similar to transfer learning: a model trained on a large dataset
# (e.g., ImageNet for image classification) transfers its learned knowledge to a similar task, 
# like medical image classification. The model doesn't start from scratch; it builds on what it 
# has already learned, adapting quickly to the new task. This saves both time and computational resources.

# Sentiment analysis is a common NLP task where transfer learning can be applied. For example,
#"This product is absolutely amazing!"	positive
#"Very poor quality. Waste of money."	negative
#"It's decent, but not worth the price."	neutral
#"I love it! Will definitely buy again."	positive
#"Worst experience ever, very disappointed."	negative
#"The product is okay, nothing special."	neutral

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

# Check if CUDA is available
print(torch.cuda.is_available())  # Should return True if a GPU is available

# Load the model and tokenizer
model_name = 'distilbert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a sentiment analysis dataset
dataset = load_dataset("sst2")  # Stanford Sentiment Treebank (binary sentiment classification)

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(
        examples['sentence'], 
        padding='max_length',
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set the training arguments
training_args = TrainingArguments(
    output_dir='./results', evaluation_strategy='epoch', num_train_epochs=3, 
    learning_rate=2e-5, per_device_train_batch_size=16, weight_decay=0.015
)

# Setup the trainer
trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['validation']  # Using 'validation' instead of 'test'
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Evaluate the model
results = trainer.evaluate()
print(results)

# Sklearn classification report
predictions = trainer.predict(tokenized_datasets["validation"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_datasets["validation"]["label"]
print(classification_report(y_true, y_pred))
