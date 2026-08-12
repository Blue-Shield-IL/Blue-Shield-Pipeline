from pydantic import BaseModel, Field

class PostAnalysis(BaseModel):
    antisemitism_score: float = Field(
        description="A score between 0.0 and 1.0 indicating how much the post contains hate-speech, "
                    "antisemitism, or anti-Zionism. General neutral mentions of Jews or Israel score low. "
                    "Only hateful, conspiratorial, or anti-Zionist content scores high — regardless of "
                    "whether the author endorses or condemns it."
    )
    ihra_labels: list[str] = Field(
        description="List of applicable IHRA antisemitism labels."
    )
    keywords: list[str] = Field(
        description="List of applicable keyword labels (e.g., 'October 7th Hamas attack', 'blood libel')."
    )
    sentiment: str = Field(
        description="The author's own opinion/stance toward Jews/Israel as expressed in the text. One of: "
                    "Supportive (pro-Jewish/pro-Israel or condemning antisemitism), Neutral (no clear stance, "
                    "e.g. pure news reporting), Negative (critical or dismissive), Hostile (antisemitic, "
                    "endorses violence, or promotes hateful tropes)."
    )
    country_of_origin: str | None = Field(
        default=None,
        description="The country or location mentioned as the origin/target if applicable. Null if none."
    )
