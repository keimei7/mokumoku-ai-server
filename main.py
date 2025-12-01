# main.py
from enum import Enum
from typing import Optional
from datetime import datetime
import os

from fastapi import FastAPI
from pydantic import BaseModel

# OpenAI クライアント
from openai import OpenAI


# ============================
#  API KEY 読み込み（Railway対応）
# ============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Railway が key を渡せていない場合はクラッシュして原因をログに出す
    raise RuntimeError("AI server: OPENAI_API_KEY is not set in environment")

print("DEBUG: OPENAI_API_KEY loaded =", bool(OPENAI_API_KEY))

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================
# FastAPI
# ============================
app = FastAPI(title="MokuMoku AI Server", version="1.0")


@app.get("/")
def root():
    return {"status": "ok"}


# ============================
# モデル
# ============================
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
    target_date: str   # yyyy-MM-dd
    locale: str = "ja-JP"
    summary: AISummary


class AICommentResponse(BaseModel):
    id: str
    scope: Scope
    target_date: str
    text: str
    generated_at: datetime


# ============================
# システムプロンプト
# ============================
SYSTEM_PROMPT = """
あなたは頭痛・体調ログアプリ「もくもくスタンプカレンダー」の専用アシスタントです。
出力は必ず **日本語** で、1〜2 文の短いコメントだけにしてください。

【注意】
- 医療的な診断・治療の提案は禁止
- 軽い振り返り・やさしいトーン
- 数値をそのまま並べず「傾向」を自然に表現
"""


# ============================
# AI コメント API
# ============================
@app.post("/v1/ai_comment", response_model=AICommentResponse)
async def ai_comment(req: AICommentRequest) -> AICommentResponse:

    scope_label = {
        Scope.day: "この日",
        Scope.week: "この週",
        Scope.month: "この月",
    }.get(req.scope, "この期間")

    parts = []

    # 気圧
    if req.summary.pressure is not None:
        p = req.summary.pressure
        parts.append(
            f"- 気圧: 最低 {p.min}, 最高 {p.max}, 変動幅 {p.delta}"
        )
    else:
        parts.append("- 気圧: 情報なし")

    # 頭痛
    h = req.summary.headache
    parts.append(
        f"- 頭痛: 合計 {h.count} 件, 最大レベル {h.max_level}, 平均 {h.avg_level}"
    )

    # 睡眠
    if req.summary.sleep_hours_avg is not None:
        parts.append(f"- 睡眠: 平均 {req.summary.sleep_hours_avg} 時間")
    else:
        parts.append("- 睡眠: 情報なし")

    # 活動
    if req.summary.activity_score_avg is not None:
        parts.append(f"- 活動量: 平均スコア {req.summary.activity_score_avg}")
    else:
        parts.append("- 活動量: 情報なし")

    stats = "\n".join(parts)

    # ユーザープロンプト
    user_prompt = f"""
{scope_label}（{req.target_date}）の記録は次の通りです。

{stats}

この内容をもとに、{scope_label}の様子を
やさしく 1〜2 文でまとめてください。
"""

    # OpenAI 呼び出し
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    text = completion.choices[0].message.content.strip()

    return AICommentResponse(
        id=f"cmt_{req.user_id}_{req.target_date}_{req.scope}",
        scope=req.scope,
        target_date=req.target_date,
        text=text,
        generated_at=datetime.utcnow(),
    )


# ============================
# Run
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
