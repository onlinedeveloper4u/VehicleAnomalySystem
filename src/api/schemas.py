from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class SensorData(BaseModel):
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

class ModelDetail(BaseModel):
    score: List[float]
    is_anomaly: List[bool]

class PredictionResponse(BaseModel):
    is_anomaly: List[bool]
    votes: List[int]
    details: Dict[str, ModelDetail]

class ThresholdConfig(BaseModel):
    isolation_forest: Optional[float] = None
    one_class_svm: Optional[float] = None
    autoencoder: Optional[float] = None
