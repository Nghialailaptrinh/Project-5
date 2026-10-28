import copy

import torch
from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"

        while True:
            mistakes = 0

            for batch in dataloader:
                x = batch['x'].squeeze(0)
                y = batch['label'].item()

                prediction = model.get_prediction(x)

                if prediction != y:
                    model.w += y * x
                    mistakes += 1

            if mistakes == 0:
                break

def train_regression(model, dataset):
    """
    Trains the regression model.
    """
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    max_epochs = 2000

    for epoch in range(max_epochs):
        total_loss = 0.0

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()

            prediction = model(x)
            loss = regression_loss(prediction, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        if average_loss <= 0.01:
            break

def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    max_epochs = 20
    target_accuracy = 0.98

    for epoch in range(max_epochs):
        model.train()

        total_loss = 0.0

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()

            logits = model(x)
            loss = digitclassifier_loss(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        validation_accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1}: "
            f"loss={average_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy >= target_accuracy:
            break

def train_languageid(model, dataset):
    """
    Trains the language identification model.
    """

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    max_epochs = 40
    target_accuracy = 0.81

    best_accuracy = 0.0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch["x"]
            x = x.movedim(1, 0)  # (batch, sequence, chars) -> (sequence, batch, chars)
            y = batch["label"]

            optimizer.zero_grad()
            logits = model(x)
            loss = languageid_loss(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            validation_accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss / len(dataloader):.4f}, "
            f"validation={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: param.detach().clone()
                for name, param in model.state_dict().items()
            }

        if validation_accuracy >= target_accuracy:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_accuracy:.2%}")


def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
