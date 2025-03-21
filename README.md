# deep-learning-challenge
COLAB: https://colab.research.google.com/drive/1i1m8Vb9hGY_TGnclmHZpSa04dXmKxlcd?usp=sharing

Target Variable(s):

The target variable for the model is IS_SUCCESSFUL, which indicates whether an organization was successful in obtaining funding. This is a binary classification problem, where 1 indicates success and 0 indicates failure.
Feature Variables(s):

The features for the model include the following variables:
APPLICATION_TYPE
AFFILIATION
CLASSIFICATION
USE_CASE
ORGANIZATION
STATUS
INCOME_AMT
SPECIAL_CONSIDERATIONS
ASK_AMT
Variables to Remove:

EIN and NAME should be removed from the input data as they are identifiers, not relevant to the prediction of the target variable. These columns do not provide useful features for the model to learn from.
Compiling, Training, and Evaluating the Model
Neurons, Layers, and Activation Functions:

The deep learning model consists of the following layers and neurons:
Input layer: Corresponds to the number of features in the data (after one-hot encoding).
First hidden layer: 80 neurons with ReLU activation function, allowing the model to learn non-linear relationships.
Second hidden layer: 30 neurons with ReLU activation function, further reducing dimensionality and introducing more complexity.
Output layer: A single neuron with Sigmoid activation function, as it is a binary classification problem (outputting a probability between 0 and 1).
Reason for Selection:
ReLU activation was chosen for the hidden layers because it performs well with deep networks and prevents the vanishing gradient problem.
Sigmoid activation for the output layer is standard for binary classification, as it outputs a probability that can be thresholded to predict the class label.
Target Model Performance:

The target model performance was to achieve a good accuracy for classifying the success of organizations in securing funding. Although the model reached reasonable accuracy, the performance was not perfect and improvements could be made.
Steps Taken to Increase Model Performance:

Hyperparameter Tuning: Adjusted the number of layers and neurons to find an optimal configuration.
Data Preprocessing: Categorical variables were one-hot encoded, and unnecessary columns were removed to ensure that only relevant features were included.
Model Optimization: Tried different optimizers (Adam) and loss functions (binary crossentropy) to improve training efficiency and convergence.
Regularization: Considered adding dropout layers to prevent overfitting, especially since the model was trained on a relatively small dataset.
Epochs and Batch Size Adjustments: The number of epochs and batch size were fine-tuned to ensure that the model had enough time to learn from the data without overfitting.
