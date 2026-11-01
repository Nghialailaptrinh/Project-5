from torch import no_grad
from torch.utils.data import DataLoader

"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import (
    regression_loss,
    digitclassifier_loss,
    languageid_loss,
    digitconvolution_Loss,
)
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
        mistakes = 1
        while mistakes:
            mistakes = 0
            for batch in dataloader:
                x = batch['x']
                label = batch['label']
                prediction = model.get_prediction(x)
                actual = label.item()
                if prediction != actual:
                    mistakes += 1
                    model.w += label * x


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on

    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1500):
        total_loss = 0.0
        for batch in dataloader:
            x = batch['x']
            y = batch['label']

            optimizer.zero_grad()

            y_predict = model(x)
            loss = regression_loss(y_predict, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

        if avg_loss <= 0.01:
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

    num_epochs = 30

    for epoch in range(num_epochs):

        model.train()

        for batch in dataloader:

            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()

            logits = model(x)

            loss = digitclassifier_loss(logits, y)

            loss.backward()

            optimizer.step()
    


def train_languageid(model, dataset):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch["x"].movedim(0, 1) 
            y = batch["label"]

            optimizer.zero_grad()
            y_pred = model(x)
            loss = languageid_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        model.eval()
        with no_grad():
            validation_accuracy = dataset.get_validation_accuracy()

        print(f"epoch {epoch + 1}: loss {average_loss:.4f}, val acc {validation_accuracy:.3f}")
        if validation_accuracy >= 0.85:
            break


def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
