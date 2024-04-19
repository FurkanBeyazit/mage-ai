import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from furkan.utils.variables import (
    X_COLS,
    Y_COLS,
     )

X_COLS = [
    "Age",
    "Fare",
    "Parch",
    "Pclass",
    "SibSp",
]
Y_COLS = [
    "Survived",
]

# if "custom" not in globals():
#     from mage_ai.data_preparation.decorators import custom
# if "test" not in globals():
#     from mage_ai.data_preparation.decorators import test

def _model_save(rf_model):
    """
    Save RandomForest model to a file.
    """
    # Here you would implement the logic to save your trained model to a file
    # Example:
    # with open('random_forest_model.pkl', 'wb') as file:
    #     pickle.dump(rf_model, file)
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
    x_train = df[X_COLS]
    y_train = df[Y_COLS].values.ravel() # RandomForest expects a 1D array for y

    # Initialize the Random Forest Classifier
    rf_model = RandomForestClassifier()

    # Train the model
    rf_model.fit(x_train, y_train)

    # Predict using the trained model
    _pred = rf_model.predict(x_train)

    # Optionally save the model
    _model_save(rf_model)

    # Assign predictions to a new column in the dataframe
    df = df.assign(Survived_predict=_pred)
    return df

# @test
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