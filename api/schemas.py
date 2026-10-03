"""Pydantic request/response models. Invalid input is rejected with HTTP 422 before it can
reach the model."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 45, "sex": 2, "height": 170, "weight": 78,
                "systolic_bp": 135, "diastolic_bp": 85,
                "cholesterol": 1, "glucose": 1, "smoking": 0, "alcohol": 0, "active": 1,
            }
        },
    )

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[1, 2] = Field(..., description="1 = female, 2 = male (dataset coding)")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=30, le=300, description="Weight in kg")
    systolic_bp: int = Field(..., ge=70, le=250, description="Systolic blood pressure (mmHg)")
    diastolic_bp: int = Field(..., ge=40, le=150, description="Diastolic blood pressure (mmHg)")
    cholesterol: Literal[1, 2, 3] = Field(..., description="1 normal, 2 above normal, 3 well above normal")
    glucose: Literal[1, 2, 3] = Field(..., description="1 normal, 2 above normal, 3 well above normal")
    smoking: Literal[0, 1] = Field(..., description="1 = smoker")
    alcohol: Literal[0, 1] = Field(..., description="1 = regular alcohol intake")
    active: Literal[0, 1] = Field(..., description="1 = physically active")

    @model_validator(mode="after")
    def systolic_above_diastolic(self):
        if self.systolic_bp <= self.diastolic_bp:
            raise ValueError("systolic_bp must be greater than diastolic_bp")
        return self

    def to_model_input(self) -> dict:
        """Map API field names onto the column names the trained pipeline expects."""
        data = self.model_dump()
        data["age_years"] = data.pop("age")
        return data


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = higher-risk profile, 0 = lower-risk profile")
    risk_category: str
    probability: float = Field(..., ge=0, le=1, description="Model output probability for class 1")
    model_version: str
    disclaimer: str = (
        "Educational portfolio project. Not a medical device and not a diagnosis."
    )
