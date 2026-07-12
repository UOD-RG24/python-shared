from pydantic import BaseModel, Field

class MatchAllSamplesExistOnAllOmicsLayersModel(BaseModel):
    dataset_id: str = Field(
        alias="datasetId",
        min_length=1,
    )

class MatchAllSamplesHaveOmicsDataModel(BaseModel):
    dataset_id: str = Field(alias="datasetId")
    status: str