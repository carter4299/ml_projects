"""
originally jupyter notebook
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier


#drive.mount('/content/drive')
file_name = '/content/drive/MyDrive/Colab Notebooks/Wine_Quality_Data.csv'
with open(file_name, 'r') as file:
    df = pd.read_csv(file_name)
num_rows, num_columns = df.shape
print(f"Dataset \"{file_name}\", has {num_rows} rows and {num_columns} columns.\n")

if df.isnull().any().any():
    print(f"Columns with null values: {df.isnull().any()}\n")
else:
    print("No columns have null values.\n")

target_label = df['quality']
label_counts = target_label.value_counts(normalize=True) * 100
print(f"Label counts:\n{label_counts}\nTop 3 quality scores by count:\n{label_counts.nlargest(3)}\n")

correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
corr_pairs = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()

print(f"Pairs of columns with highest positive correlation:\n{corr_pairs[corr_pairs != 1].head(1)}\n\nPairs of columns with highest negative correlation:\n{corr_pairs.tail(1)}\n")

df = df.drop('color', axis=1)

y = target_label
X = df.drop('quality', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean of X_scaled:\n {X_scaled.mean(axis=0)}\nStandard deviation of X_scaled:\n {X_scaled.std(axis=0)}\n")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

def onehot(y):
    """
    Converts an array of labels to one-hot encoding.

    Parameters:
    -----------
    y : numpy.ndarray
        An array of labels. It must have a shape of (n_samples, ).

    Returns:
    --------
    numpy.ndarray
        A matrix of one-hot encoding. It has a shape of (n_samples, n_classes).
    """

    y_array = y.to_numpy() if isinstance(y, pd.Series) else y
    n_samples = y_array.shape[0]
    unique_labels = np.unique(y_array)
    n_classes = len(unique_labels)
    one_hot = np.zeros((n_samples, n_classes))

    for i, label in enumerate(y_array):
        one_hot[i, np.where(unique_labels == label)] = 1

    return one_hot


def cross_entropy_loss(y_true, y_pred):
    """
    Computes the cross-entropy loss between the true labels and predicted labels.

    Parameters
    ----------
    y_true : numpy array
        Array of true labels with shape (m, n_classes).
    y_pred : numpy array
        Array of predicted labels with shape (m, n_classes).

    Returns
    -------
    float
        Cross-entropy loss between y_true and y_pred.

    Notes
    -----
    This function assumes that the labels are one-hot encoded.
    """

    N = y_true.shape[0]
    ce_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / N
    return ce_loss


def softmax(scores):
    """
    Compute the softmax of the given scores.

    Parameters:
    -----------
    scores : numpy.ndarray
        A 2D numpy array of shape (m, n), where m is the number of samples and n is the number of classes.

    Returns:
    --------
    probs : numpy.ndarray
        A 2D numpy array of shape (m, n) containing the probabilities of each sample belonging to each class.
    """

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def logistic_regression(X_train, y_train):
    """
    Performs logistic regression using softmax activation and gradient descent optimizer to classify the input data.

    Parameters:
    -----------
    X_train : numpy.ndarray
        The input training data of shape (num_samples, num_features).
    y_train : numpy.ndarray
        The training labels of shape (num_samples,).

    Returns:
    --------
    W : numpy.ndarray
        The learned weights of shape (num_features, num_classes).
    b : numpy.ndarray
        The learned bias of shape (1, num_classes).
    loss_list : list
        The list of loss values at each epoch during training.
    """

    # get the number of samples and features from X_train (2pts)
    num_samples, num_features = X_train.shape

    # convert training labels to one-hot encoded labels (2pts)
    y_train_onehot = onehot(y_train)

    # get the number of target classes from y_train (2pts)
    num_classes = y_train_onehot.shape[1]

    # initialize the weights and bias with numpy arrays of zeros (1+1 = 2pts)
    W = np.zeros((num_features, num_classes))
    b = np.zeros((1, num_classes))

    # set hyperparameters (1+1 = 2pts)
    learning_rate = 0.01
    max_epochs = 1000
    ## set the max number of epochs you want to train for
    max_epochs = 1000

    ## initialize a list to store the loss values (1pt)
    loss_list = []

    '''
    Write a for loop over epochs.
    In each epoch:
        compute the score for each class, 
        compute the softmax probabilities, 
        compute the cross-entropy loss, 
        compute the gradients of the loss with respect to the weights and bias, 
        update the weights and bias using the gradients and the learning rate.
    '''
    # (9pts)
    for epoch in range(max_epochs):
        # compute the score (Z) for each class.
        Z = np.dot(X_train, W) + b

        # calculate the softmax probabilities
        probabilities = softmax(Z)

        # compute the cross-entropy loss
        loss = cross_entropy_loss(y_train_onehot, probabilities)

        # compute the gradients of the loss with respect to the weights and bias
        grad_W = (-1 / num_samples) * np.dot(X_train.T, (y_train_onehot - probabilities))
        grad_b = (-1 / num_samples) * np.sum(y_train_onehot - probabilities, axis=0)

        # update the weights and bias using the gradients and the learning rate
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        # For tracking progress, print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            loss_list.append(loss)

    return W, b, loss_list


# train the model
W, b, loss_list = logistic_regression(X_train, y_train)

# plot the loss curve
plt.figure()
plt.plot(np.arange(len(loss_list)) * 100, loss_list)
plt.xlabel("Epochs")
plt.ylabel("Cross-entropy Loss")
plt.title("Training Loss Curve")
plt.show()


def predict(X_test, W, b):
    '''
    X_test: a numpy array of testing features
    W: a numpy array of weights
    b: a numpy array of bias
    return: a numpy array of one-hot encoded labels
    '''
    # compute the scores
    scores = np.dot(X_test, W) + b

    # compute the probabilities
    probs = softmax(scores)

    # get the predicted labels
    predicted_labels = np.argmax(probs, axis=1)
    predicted_labels += np.min(y)

    # return the predicted labels
    return predicted_labels


y_pred = predict(X_test, W, b)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

logreg_cv = LogisticRegressionCV(cv=5, solver='lbfgs', max_iter=1000, multi_class='multinomial')

logreg_cv.fit(X_train, y_train)

y_pred = logreg_cv.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy(CV): {accuracy:.4f}")
print(f"Precision(CV): {precision:.4f}")
print(f"Recall(CV): {recall:.4f}")
print(f"F1 Score(CV): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels(CV)')
plt.ylabel('True Labels(CV)')
plt.title('Confusion Matrix Heatmap(CV)')
plt.show()

print("\n\tThey both had similar results, however the log reg CV slightly outperformed my log reg model\n")

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,),
                               activation='relu',
                               solver='sgd',
                               learning_rate_init=0.01,
                               max_iter=1000,
                               random_state=42)

mlp_classifier.fit(X_train, y_train)
y_pred_mlp = mlp_classifier.predict(X_test)

plt.figure()
plt.plot(mlp_classifier.loss_curve_)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP Loss Curve")
plt.show()

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print(f"Accuracy(MLP): {accuracy_mlp:.4f}")
print(f"Precision(MLP): {precision_mlp:.4f}")
print(f"Recall(MLP): {recall_mlp:.4f}")
print(f"F1 Score(MLP): {f1_mlp:.4f}")

cm_mlp = confusion_matrix(y_test, y_pred_mlp)

sns.heatmap(cm_mlp, annot=True, fmt='d', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels(MLP)')
plt.ylabel('True Labels(MLP)')
plt.title('Confusion Matrix Heatmap(MLP)')
plt.show()

print("\n\tThe MLP had slightly better results than the log reg cv, which outperformed the original model, making it the best one so far.\n")

mlp_classifier_2hl = MLPClassifier(hidden_layer_sizes=(100, 100),
                                 activation='relu',
                                 solver='adam',
                                 learning_rate_init=0.01,
                                 max_iter=1000,
                                 random_state=42)

mlp_classifier_2hl.fit(X_train, y_train)

y_pred_mlp_2hl = mlp_classifier_2hl.predict(X_test)

plt.figure()
plt.plot(mlp_classifier_2hl.loss_curve_)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

accuracy_mlp_2hl = accuracy_score(y_test, y_pred_mlp_2hl)
precision_mlp_2hl = precision_score(y_test, y_pred_mlp_2hl, average='weighted')
recall_mlp_2hl = recall_score(y_test, y_pred_mlp_2hl, average='weighted')
f1_mlp_2hl = f1_score(y_test, y_pred_mlp_2hl, average='weighted')

print(f"Accuracy(MLP_2HL): {accuracy_mlp_2hl:.4f}")
print(f"Precision(MLP_2HL): {precision_mlp_2hl:.4f}")
print(f"Recall(MLP_2HL): {recall_mlp_2hl:.4f}")
print(f"F1 Score(MLP_2HL): {f1_mlp_2hl:.4f}")

cm_mlp_2hl = confusion_matrix(y_test, y_pred_mlp_2hl)

sns.heatmap(cm_mlp_2hl, annot=True, fmt='d', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Labels(MLP_2)')
plt.ylabel('True Labels(MLP_2)')
plt.title('Confusion Matrix Heatmap(MLP_2)')
plt.show()

print("\n\tThe MLPC with two hidden layer's loss curve starts at 0 epochs with a lower loss(1.2 compared to the original 1.8). It also reaches a value of 0.2 in only 120 epochs, compared to original reach 0.8 in 500.")

print("\n\tThe MLPC with two hidden layer's performed the best by a pretty significant margin. The largest difference in the models before was ~0.04 in F1 score(CV->MLP). The smallest increase on this model was larger than that.")












