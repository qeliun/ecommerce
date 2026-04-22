import pandas as pd
from catboost import CatBoostClassifier
import json

def load_model(model_path='ecom_model.cbm'):
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def predict_purchase(model, input_data):
    df_input = pd.DataFrame([input_data])
    
    proba = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]
    
    return {
        "purchase_probability": round(float(proba), 4),
        "will_purchase": bool(prediction)
    }

if __name__ == "__main__":
   
    sample_session = {
        "Administrative": 2,
        "Administrative_Duration": 50.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 200.5,
        "BounceRates": 0.01,
        "ExitRates": 0.02,
        "PageValues": 15.5, 
        "SpecialDay": 0.0,
        "Month": "May",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False
    }

    trained_model = load_model()

    result = predict_purchase(trained_model, sample_session)

    print("\nResult:")
    print(json.dumps(result, indent=4))