from pydantic import BaseModel, Field
from typing import List, Optional


class SensorData(BaseModel):
    """
    Input schema for NASA CMAPSS turbofan engine sensor readings.
    
    Columns correspond to:
    - engine_id: Unit number
    - cycle: Time in cycles
    - setting1-3: Operational settings
    - s1-s21: Sensor measurements
    
    Reference: Saxena et al., "Damage Propagation Modeling for Aircraft Engine 
    Run-to-Failure Simulation", PHM08
    """
    engine_id: int = Field(..., description="Engine unit number")
    cycle: int = Field(..., description="Current operational cycle")
    
    # Operational settings
    setting1: float = Field(..., description="Operational setting 1")
    setting2: float = Field(..., description="Operational setting 2")
    setting3: float = Field(..., description="Operational setting 3")
    
    # Sensor measurements (s1-s21)
    s1: float = Field(default=0.0, description="Sensor 1 - (T2) Total temperature at fan inlet")
    s2: float = Field(..., description="Sensor 2 - (T24) Total temperature at LPC outlet")
    s3: float = Field(..., description="Sensor 3 - (T30) Total temperature at HPC outlet")
    s4: float = Field(..., description="Sensor 4 - (T50) Total temperature at LPT outlet")
    s5: float = Field(default=0.0, description="Sensor 5 - (P2) Pressure at fan inlet")
    s6: float = Field(default=0.0, description="Sensor 6 - (P15) Total pressure in bypass-duct")
    s7: float = Field(..., description="Sensor 7 - (P30) Total pressure at HPC outlet")
    s8: float = Field(..., description="Sensor 8 - (Nf) Physical fan speed")
    s9: float = Field(..., description="Sensor 9 - (Nc) Physical core speed")
    s10: float = Field(default=0.0, description="Sensor 10 - (epr) Engine pressure ratio")
    s11: float = Field(..., description="Sensor 11 - (Ps30) Static pressure at HPC outlet")
    s12: float = Field(..., description="Sensor 12 - (phi) Ratio of fuel flow to Ps30")
    s13: float = Field(..., description="Sensor 13 - (NRf) Corrected fan speed")
    s14: float = Field(..., description="Sensor 14 - (NRc) Corrected core speed")
    s15: float = Field(..., description="Sensor 15 - (BPR) Bypass Ratio")
    s16: float = Field(default=0.0, description="Sensor 16 - (farB) Burner fuel-air ratio")
    s17: float = Field(..., description="Sensor 17 - (htBleed) Bleed Enthalpy")
    s18: float = Field(default=0.0, description="Sensor 18 - (Nf_dmd) Demanded fan speed")
    s19: float = Field(default=0.0, description="Sensor 19 - (PCNfR_dmd) Demanded corrected fan speed")
    s20: float = Field(..., description="Sensor 20 - (W31) HPT coolant bleed")
    s21: float = Field(..., description="Sensor 21 - (W32) LPT coolant bleed")


class PredictionResponse(BaseModel):
    """Response schema for anomaly predictions."""
    is_anomaly: List[bool] = Field(description="Anomaly flag for each input record")
    anomaly_types: List[str] = Field(description="Type of anomaly detected (Normal, Degraded)")
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


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions from file uploads."""
    data: List[SensorData]
