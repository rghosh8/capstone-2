from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def eval(model, train, test, test_size):
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=test_size, random_state=32)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1}')
    plot_confusion_matrix(model, X_test, y_test, normalize='pred')
    plt.show()

    
    return None
 