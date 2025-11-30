from enum import Enum
from typing import Optional
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="MokuMoku AI Server", version="0.1.0")


class Scope(str, Enum):
    day = "day"
    week = "week"
    month = "month"


class PressureSummary(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    delta: Optional[float] = None


class HeadacheSummary(BaseModel):
    count: int
    max_level: int
    avg_level: Optional[float] = None
    first_at: Optional[str] = None
    last_at: Optional[str] = None


class AISummary(BaseModel):
    pressure: Optional[PressureSummary] = None
    headache: HeadacheSummary
    sleep_hours_avg: Optional[float] = None
    activity_score_avg: Optional[float] = None


class AICommentRequest(BaseModel):
    user_id: str
    scope: Scope
    target_date: str  # "yyyy-MM-dd"
    locale: str = "ja-JP"
    summary: AISummary


class AICommentResponse(BaseModel):
    id: str
    scope: Scope
    target_date: str
    text: str
    generated_at: datetime


@app.post("/v1/ai_comment", response_model=AICommentResponse)
async def ai_comment(req: AICommentRequest) -> AICommentResponse:
    """
    ここで本当は LLM を呼び出してコメント生成する。
    今は scope 別にそれっぽい文を返すダミー実装。
    """

    if req.scope == Scope.day:
        text = (
            f"{req.target_date} は、頭痛の回数が {req.summary.headache.count} 回でした。"
            "今日は少ししんどかったかもしれませんね。"
            "無理のないペースで過ごせていれば、それだけで十分えらい一日です。"
        )
    elif req.scope == Scope.week:
        text = (
            f"{req.target_date} を含む1週間は、全体として頭痛が出やすい日と"
            "落ち着いた日の差がありました。"
            "特に、気圧の変化が大きい日ほど痛みが強くなりやすい傾向がありそうです。"
        )
    else:  # month
        text = (
            f"{req.target_date} を含む1か月は、頭痛の日数や強さのパターンが"
            "少しずつ見えてきています。"
            "睡眠時間や活動量との関係も、今後もう少し観察していくと、"
            "自分なりのクセが掴めてきそうです。"
        )

    return AICommentResponse(
        id=f"cmnt_{req.target_date}_{req.scope}_{req.user_id}",
        scope=req.scope,
        target_date=req.target_date,
        text=text,
        generated_at=datetime.utcnow(),
    )
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
@app.get("/")
def root():
    return {"status": "ok"}
    
    
    
