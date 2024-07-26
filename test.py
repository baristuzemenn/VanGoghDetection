from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


from utils import get_test_data, print_baseline


def evaluate_model(model, testx, testy):


    # Get model score
    score = model.evaluate(testx, testy, batch_size=32)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])

    # Get predictions
    predictions = model.predict(testx)
    # If your model ends with a sigmoid activation, this step will convert probabilities to binary predictions.
    predictions_binary = (predictions > 0.5).astype(int)

    # Calculate precision, recall, and F1 score
    precision = precision_score(testy, predictions_binary)
    recall = recall_score(testy, predictions_binary)
    f1 = f1_score(testy, predictions_binary)

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)

    # Print classification report
    print(classification_report(testy, predictions_binary))

    # Print confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(testy, predictions_binary))


def generate_predictions(model, testx, outfile):

    probabilities = model.predict(testx)
    predictions = []
    for prob in probabilities:
        if prob < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    np.savetxt(outfile, predictions, fmt='%d')


def main():


    # Load in test data and model
    testx, testy = get_test_data()
    model = load_model('model.ker')

    # Evaluate test accuracy
    evaluate_model(model, testx, testy)

    # Generate predictions
    generate_predictions(model, testx, 'test-yhat')


if __name__ == '__main__':
    main()





