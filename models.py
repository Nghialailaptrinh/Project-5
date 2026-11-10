from torch import stack, tril
from torch.nn import Module

"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()

        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return (x @ self.w.T).reshape(1)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self(x)

        "*** YOUR CODE HERE ***"
        return 1 if score.item() >= 0 else -1


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        self.linear1 = Linear(1, 64)
        self.linear2 = Linear(64, 64)
        self.linear3 = Linear(64, 1)
        

        self.linear1 = Linear(1, 64)
        self.linear2 = Linear(64, 64)
        self.linear3 = Linear(64, 1)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        hidden1 = relu(self.linear1(x))
        hidden2 = relu(self.linear2(hidden1))
        return self.linear3(hidden2)


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.linear1 = Linear(784, 256)
        self.linear2 = Linear(256, 128)
        self.linear3 = Linear(128, 10)

    def forward(self, x):
        
        hidden1 = relu(self.linear1(x))
        hidden2 = relu(self.linear2(hidden1))
        logits = self.linear3(hidden2)





        hidden1 = relu(self.linear1(x))
        hidden2 = relu(self.linear2(hidden1))
        logits = self.linear3(hidden2)

        return logits


class LanguageIDModel(Module):
    """
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super().__init__()

        self.hidden_size = 256

        self.input_layer = Linear(
            self.num_chars,
            self.hidden_size
        )

        self.hidden_layer = Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False
        )

        self.output_layer = Linear(
            self.hidden_size,
            len(self.languages)
        )

    def forward(self, xs):


        batch_size = xs[0].shape[0]

        hidden = zeros(batch_size, self.hidden_size)

        for char in xs:
            char_features = self.input_layer(char)
            hidden = relu(char_features + self.hidden_layer(hidden))

        logits = self.output_layer(hidden)
        return logits


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    "*** YOUR CODE HERE ***"

    input_h, input_w = input_tensor_dimensions
    weight_h, weight_w = weight_dimensions
    out_h = input_h - weight_h + 1
    out_w = input_w - weight_w + 1

    Output_Tensor = zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Output_Tensor[i, j] = tensordot(
                input[i:i + weight_h, j:j + weight_w], weight, dims=2
            )

    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        # After 3x3 convolution on 28x28 image → output 26x26 = 676 features
        self.linear1 = Linear(676, 256)
        self.linear2 = Linear(256, output_size)

    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = relu(self.linear1(x))
        return self.linear2(x)


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""

        K = self.k_layer(input) 
        Q = self.q_layer(input)  
        V = self.v_layer(input)  



        scale = self.layer_size ** 0.5
        weights = matmul(Q, movedim(K, -1, -2)) / scale  

  
        weights = weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]

        weights = softmax(weights, dim=-1)  

  
        output = matmul(weights, V)

        return output
