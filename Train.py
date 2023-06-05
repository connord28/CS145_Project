import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from datetime import datetime


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, loss_fn, device
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
        device:          Device that we're using
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    # loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    # writer = torch.utils.tensorboard.SummaryWriter(summary_path)
    
    model.train()
    step = 0
    best_val_accuracy = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            input_data, labels = batch
            input_data, labels = input_data.to(device), labels.to(device)
            optimizer.zero_grad()

            predictions = model(input_data)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()


            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:

                train_loss, train_accuracy = evaluate(train_loader, model, loss_fn, device)
                # writer.add_scalar("train_accuracy", train_accuracy, global_step = step)
                # writer.add_scalar("train_loss", loss, global_step = step)

                val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device)
                # writer.add_scalar("val_loss", val_loss, global_step=step)
                # writer.add_scalar("val_accuracy", val_accuracy, global_step=step)
                
                print(f"Eval:\t{step/n_eval}")
                print(f"Train loss:\t{train_loss}")
                print(f"Train Accuracy:\t{train_accuracy}")
                print(f"Validation loss:\t{val_loss}")
                print(f"Validation Accuracy:\t{val_accuracy}")
            
            step += 1
        
        # save model if it's better than best model in the training loop
        if val_accuracy > best_val_accuracy:
            print(f'best model accuracy increased from {best_val_accuracy} to {val_accuracy}')
            best_val_accuracy = val_accuracy
            model_name = str(model).split('\n')[0]
            print(f'saving {model_name}...')
            with open(f'{model_name}.txt', 'w') as f:
                f.write(str(model))
            f.close()
            torch.save(model.state_dict(), f'{model_name}.pt')
            

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

### Used for evaluating performance on validation or test set
def evaluate(eval_loader, model, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(eval_loader):
        input_data, labels = batch
        input_data, labels = input_data.to(device), labels.to(device)
        predictions = model(input_data)
        total_loss += loss_fn(predictions, labels).item()
        correct += (predictions.argmax(axis=1) == labels).sum().item()
        total += len(labels)    
    
    loss = total_loss / total
    accuracy = correct / total

    model.train()
    
    return loss, accuracy

### Performs evaluation using an ensemble of models
def evaluateEnsemble(eval_loader, models, loss_fn, device):
    for model in models:
      model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(eval_loader):
        input_data, labels = batch
        input_data, labels = input_data.to(device), labels.to(device)
        
        predictions = models[0](input_data)
        for i in range(1, len(models)):
          predictions += models[i](input_data)
        predictions /= len(models)
        total_loss += loss_fn(predictions, labels).item()
        correct += (predictions.argmax(axis=1) == labels).sum().item()
        total += len(labels)    
    
    loss = total_loss / total
    accuracy = correct / total

    model.train()
    
    return loss, accuracy
