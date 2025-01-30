import pickle

def saveObject(objectObtained:object=None, filePath:str=None) -> None:
    """
    # Description
        -> Saves the best estimator found (passed onto the function).
    -----------------------------------------------------------------    
    Parameters:
    := param: objectObtained - Object to store in a pickle file.
    := param: filePath - File path to the estimator.
    := return: None, since we are only saving an estimator.
    """

    # Check if a estimator was provided
    if objectObtained is None:
        raise ValueError("Missing a instance of a object to save!")
    
    # Check if the path is valid
    if filePath is None:
        raise ValueError("Invalid Path provided!")

    # Save the best estimator
    with open(filePath, 'wb') as f:
        pickle.dump(objectObtained, f)

def loadObject(filePath:str=None) -> object:
    """
    # Description
        -> Loads a previously saved object.
    ----------------------------------------------
    := param: filePath - File path to the object saved.
    := return: The stored object.
    """

    # Check if the path is valid
    if filePath is None:
        raise ValueError("Invalid Path provided!")

    # Load the best estimator
    with open(filePath, 'rb') as f:
        objectObtained = pickle.load(f)

    # Return the best estimator
    return objectObtained