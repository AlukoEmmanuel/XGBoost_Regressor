import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = joblib.load("model/model.pkl")

app = FastAPI()


class InputData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    northwest: int
    southeast: int
    southwest: int


@app.post("/api/predict")
def predict(data: InputData):
    input_features = [
        data.age, data.sex, data.bmi, data.children, data.smoker,
        data.northwest, data.southeast, data.southwest
    ]
    prediction = model.predict([input_features])
    # Convert numpy types to native Python types
    predicted_charges = prediction[0].item()
    return {"predicted_charges": predicted_charges}


# Endpoint for sex values
@app.get("/api/entities/sex")
def get_sex_labels():
    return {
        "sex": [
            {"value": 1, "label": "Male"},
            {"value": 0, "label": "Female"},
        ]
    }


# Endpoint for smoker values
@app.get("/api/entities/smoker")
def get_smoker_labels():
    return {
        "smoker": [
            {"value": 1, "label": "Yes"},
            {"value": 0, "label": "No"},
        ]
    }


# Corrected endpoint for region values
@app.get("/api/entities/region")
def get_region_labels():
    return {
        "region": [
            {"columns": [1, 0, 0], "label": "Northwest"},
            {"columns": [0, 1, 0], "label": "Southeast"},
            {"columns": [0, 0, 1], "label": "Southwest"},
            {"columns": [0, 0, 0], "label": "Northeast"}
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
