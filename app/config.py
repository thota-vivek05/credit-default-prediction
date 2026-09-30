from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://cs_user:cs_pass@localhost:5432/credit_scoring"
    model_path: str = "./models/xgboost_model.pkl"
    default_threshold: float = 0.15
    low_risk_threshold: float = 0.10
    medium_risk_threshold: float = 0.25
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
