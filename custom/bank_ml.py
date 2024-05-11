from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import randint
import os
import pickle





LABEL_COLUMN = 'target'


def build_training_and_test_set(df: DataFrame) -> None:
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    return train_test_split(X, y, random_state=42)

def train_model(X, y) -> None:
    model = RandomForestClassifier()
    param_distributions ={
    'n_estimators': randint(100, 1000),
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 50),
    'min_samples_leaf': randint(1, 50),
    'max_features': randint(1,10),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes':randint(10,100)
}
    search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_distributions,
                                   n_iter=50,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1
                                   )
    fitted=search.fit(X, y)

    return fitted


def score_model(model, X, y) -> None:
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)
def class_report(model,X,y) -> None:
    y_pred = model.predict(X)
    return classification_report(y,y_pred,target_names=['No','Yes'])

@custom
def export_data(df: DataFrame) -> None:
    X_train, X_test, y_train, y_test = build_training_and_test_set(df)
    model = train_model(X_train, y_train)

    score = score_model(model, X_test, y_test)
    print(f'Accuracy: {score}')
    report_1= class_report(model,X_test,y_test)
    print(report_1)
    

    cwd = os.getcwd()
    filename = f'{cwd}/finalized_model.lib'
    print(f'Saving model to {filename}')
    pickle.dump(model, open(filename, 'wb'))

    print(f'Saving training and test set')
    X_train.to_csv(f'{cwd}/X_train')
    X_test.to_csv(f'{cwd}/X_test')
    y_train.to_csv(f'{cwd}/y_train')
    y_test.to_csv(f'{cwd}/y_test')
    
@test
def test_output(df) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'    