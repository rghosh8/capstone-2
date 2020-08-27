from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def eval(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1}')
    
    return None
 