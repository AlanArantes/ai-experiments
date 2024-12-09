import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# Custom dataset class for patient data
class PatientDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_model(model, train_dataloader, val_dataloader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")

        # Validation
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            loss = outputs.loss
            total_val_loss += loss.item()

            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        print(f"Average validation loss: {avg_val_loss}")
        print(f"Validation accuracy: {accuracy}")


def predict_priority(model, tokenizer, patient_data, device):
    model.eval()
    dataset = PatientDataset(patient_data, [0] * len(patient_data), tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)

    priorities = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            priorities.extend(outputs.logits.softmax(dim=1).cpu().numpy())

    return priorities


def main():
    # Sample patient data (replace with your actual data)
    patient_data = [
        "Severe chest pain, shortness of breath, male, age 65",
        "Mild headache, female, age 25",
        "High fever, cough, difficulty breathing, male, age 45",
        # Add more patient descriptions
    ]

    # Sample priority levels (0: low, 1: medium, 2: high)
    labels = [2, 0, 1]  # Replace with actual priority levels

    # Initialize BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # Number of priority levels
    )

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        patient_data, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = PatientDataset(X_train, y_train, tokenizer)
    val_dataset = PatientDataset(X_val, y_val, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    train_model(model, train_dataloader, val_dataloader, device)

    # Example of prioritizing new patients
    new_patients = [
        "Severe abdominal pain, vomiting, female, age 35",
        "Minor cuts and bruises, male, age 28",
    ]

    priorities = predict_priority(model, tokenizer, new_patients, device)

    # Create priority list
    priority_list = [
        (patient, priority) for patient, priority in zip(new_patients, priorities)
    ]
    priority_list.sort(key=lambda x: x[1].max(), reverse=True)

    print("\nPatient Priority List:")
    for patient, priority in priority_list:
        priority_level = ["Low", "Medium", "High"][priority.argmax()]
        print(f"Patient: {patient}")
        print(f"Priority Level: {priority_level}\n")


if __name__ == "__main__":
    main()
