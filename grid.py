from itertools import product
import pickle
from training import train_model, GPTLanguageModel, get_random_chunk
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Define hyperparameter grid
param_grid = {
    'learning_rate': [1e-4, 3e-5, 1e-5],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300]
}

# Logging results
results = []

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 10000  # Adjust based on your vocab.txt

# Run grid search
for params in product(*param_grid.values()):
    learning_rate, batch_size, epochs = params
    print(f"Testing: lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Initialize model
    model = GPTLanguageModel(vocab_size).to(device)

    # Train model and capture final training/validation loss
    trained_model, final_train_loss, final_val_loss = train_model(
        model,
        get_random_chunk,  # Your data loader logic
        lr=learning_rate,
        epochs=epochs,
        eval_iters=10,  # Set to a reasonable value
        batch_size=batch_size,
        device=device
    )

    # Save the trained model
    model_path = f"model_lr{learning_rate}_bs{batch_size}_ep{epochs}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(trained_model, f)

    print(f"Model saved to {model_path}")

    # Record results
    results.append({
        'Learning Rate': learning_rate,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Training Loss': final_train_loss,
        'Validation Loss': final_val_loss,
        'Model Path': model_path
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Output the results
print("\nGrid Search Results:")
print(results_df)

# Save results to a CSV file
results_df.to_csv('grid_search_results.csv', index=False)

# Plot Validation Loss vs Epochs for different Batch Sizes
plt.figure(figsize=(10, 6))
for batch_size in results_df['Batch Size'].unique():
    subset = results_df[results_df['Batch Size'] == batch_size]
    plt.plot(subset['Epochs'], subset['Validation Loss'], marker='o', label=f'Batch Size {batch_size}')

# Add labels and legend
plt.title('Validation Loss vs Epochs for Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Plot Training Loss vs Epochs for different Learning Rates
plt.figure(figsize=(10, 6))
for lr in results_df['Learning Rate'].unique():
    subset = results_df[results_df['Learning Rate'] == lr]
    plt.plot(subset['Epochs'], subset['Training Loss'], marker='o', label=f'Learning Rate {lr}')

# Add labels and legend
plt.title('Training Loss vs Epochs for Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.grid()
plt.show()
