from pydantic import BaseModel, Field

class PostAnalysis(BaseModel):
    antisemitism_score: float = Field(
        description="A gradient score between 0.0 and 1.0 representing the confidence that the post contains "
                    "antisemitism or justifies violence against Jews."
    )
    ihra_labels: list[str] = Field(
        description="List of applicable IHRA antisemitism labels."
    )
    keywords: list[str] = Field(
        description="List of applicable keyword labels (e.g., 'October 7th Hamas attack', 'blood libel')."
    )
    sentiment: str = Field(
        description="The overall sentiment of the text. One of: Supportive, Neutral, Negative, Hostile."
    )
    country_of_origin: str | None = Field(
        default=None,
        description="The country or location mentioned as the origin/target if applicable. Null if none."
    )
