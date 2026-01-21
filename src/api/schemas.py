from pydantic import BaseModel, Field
from typing import List, Optional


class SensorData(BaseModel):
    """Input schema for sensor readings from a vehicle."""
    Battery_Voltage: float
    Battery_Current: float
    Battery_Temperature: float
    Motor_Temperature: float
    Motor_Vibration: float
    Motor_Torque: float
    Motor_RPM: float
    Power_Consumption: float
    Brake_Pressure: float
    Tire_Pressure: float
    Tire_Temperature: float
    Suspension_Load: float
    Ambient_Temperature: float
    Ambient_Humidity: float
    Driving_Speed: float
    Vehicle_ID: Optional[str] = "default"


class PredictionResponse(BaseModel):
    """Response schema for anomaly predictions."""
    is_anomaly: List[bool] = Field(description="Anomaly flag for each input record")
    anomaly_types: List[str] = Field(description="Type of anomaly detected (Normal, Spike, Drift)")
    scores: List[float] = Field(description="Anomaly score for each record")
    thresholds: dict = Field(description="Thresholds used for classification")
    version: str = Field(description="Model version used for prediction")


class ThresholdConfig(BaseModel):
    """Schema for updating anomaly detection threshold."""
    isolation_forest: Optional[float] = Field(None, description="New threshold value")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    version: str
    threshold: float
    model_loaded: bool
