from preprocess import get_training_set
from modeling import get_model
from evaluating import evaluate

if __name__ == '__main__':
    training_set = get_training_set()
    model = get_model(training_set)
    evaluate(model)

