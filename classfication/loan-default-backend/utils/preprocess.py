import numpy as np

def preprocess_input(data):
    """
    data: dict from API Input
    returns: np.array of processed features
    """
    # Extract & process fields
    processed_data = [
        data['ApplicantIncome'],
        data['CoapplicantIncome'],
        data['LoanAmount'],
        data['Loan_Amount_Term'],
        data['Credit_History'],
        data['Gender'],
        data['Married'],
        data['Education'],
        data['Self_Employed'],
        data['Property_Area']
    ]
    return np.array(processed_data).reshape(1, -1)
