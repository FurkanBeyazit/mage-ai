import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

def _model_save(rf_model):
    """
    Save RandomForest model to a file.
    """
    # Here you would implement the logic to save your trained model to a file
    # Example:
    os.makedirs('../result', exist_ok= True)
    with open('../result/random_forest_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    pass

@custom
def random_forest_train(df: pd.DataFrame, *args, **kwargs):
    """
    Train a Random Forest Classifier and predict the 'Survived' column.
   
    Args:
        df: Data frame containing the training data.
       
    Returns:
        Data frame with a new column 'Survived_predict' with predictions.
    """
    # Prepare the data
    X = df.drop(['MedHouseVal'], axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # define the parameter space that will be searched over
    param_distributions = {'n_estimators': randint(1, 5),
                           'max_depth': randint(5, 10)}

    # now create a searchCV object and fit it to the data
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                                n_iter=10,
                                param_distributions=param_distributions,
                                random_state=0)


    # Initialize the Random Forest Classifier
    rf_model = RandomForestRegressor(random_state=0),

    # Train the model, now create a searchCV object and fit it to the data
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                                n_iter=10,
                                param_distributions=param_distributions,
                                random_state=0)

    search.fit(X_train, y_train)
    tf = pd.DataFrame(search.cv_results_)[['param_max_depth', 'param_n_estimators', 'params', 'mean_test_score', 'rank_test_score']]
    print(tf.sort_values('rank_test_score'))
    print(f"The Best model's parameters is {search.best_params_}")
    print(f'The Best accuracy score of model is {search.score(X_test, y_test)}')

    # Predict using the trained model
    y_pred = search.predict(X_test)
   
    # Optionally save the model
    _model_save(search)

    # Assign predictions to a new column in the dataframe
    tf = pd.merge(X_test, y_test, left_index=True, right_index=True)
    tf['Hosing_pred'] = y_pred
    return tf

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
   
    Args:
        output: The output from the random_forest_train function.
    """
    assert output is not None, "The output is undefined"
    assert 'Survived_predict' in output.columns, "Prediction column is missing in the output dataframe"
    # You can add more tests to check the quality of your predictions,
    # such as accuracy score, confusion matrix, etc.