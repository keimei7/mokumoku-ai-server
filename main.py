from enum import Enum
from typing import Optional
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ★ 追加
import os
from openai import OpenAI

app = FastAPI(title="MokuMoku AI Server", version="0.1.0")

# ★ ここを書き換え
api_key = (
    os.getenv("OPENAI_API_KEY")      # Railway で設定した名前
    or os.getenv("_OPENAI_API_KEY")  # 念のため、前に付けたかもしれないパターンも見る
)

if not api_key:
    # ここでわざとエラー出すと、ログに理由がハッキリ残る
    raise RuntimeError("AI server: OPENAI_API_KEY is not set in environment")

client = OpenAI(api_key=api_key)

# ★ 追加：システムプロンプト
SYSTEM_PROMPT = """
あなたは頭痛・体調ログアプリ「もくもくスタンプカレンダー」の専用アシスタントです。
出力は必ず **日本語** で、1〜2文の短いコメントだけにしてください。

【役割】
- ユーザーが「この日 / この週 / この月はどんな感じだったか」を、ざっくり振り返れるように、
  やさしいトーンの一言コメントを出します。

【入力データの意味】
- pressure.min / max / delta : その期間の気圧の最小値・最大値・変動幅(hPa)。None の場合は情報なし。
- headache.count             : 頭痛スタンプの総数
- headache.max_level         : 一番強かった頭痛レベル (数値が大きいほど強い)
- headache.avg_level         : 頭痛レベルの平均値。None の場合は情報なし。
- sleep_hours_avg            : 平均睡眠時間(時間)。None の場合は情報なし。
- activity_score_avg         : 活動量スコア平均。大きいほどよく動いているイメージ。None の場合は情報なし。

【コメント方針】
- 医療的な診断・治療・薬の提案は絶対にしない。
- 「〜かもしれませんね」「〜な傾向がありそうです」のように、あくまでゆるい振り返りにとどめる。
- 数値をそのまま列挙するのではなく、「スタンプが多い／少ない」「気圧の変動が大きかった」
  など、ユーザーがパッとイメージしやすい日本語にする。
- 重くなりすぎない、やさしい励まし系のトーンで書く。
"""

# ここでヘルスチェック用のルート
@app.get("/")
def root():
    return {"status": "ok"}


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
    LLM を呼び出して、scope・summary に応じたコメントを1〜2文生成する。
    """

    # スコープの日本語ラベル
    scope_label = {
        Scope.day: "この日",
        Scope.week: "この週",
        Scope.month: "この月",
    }.get(req.scope, "この期間")

    # ----- ユーザープロンプトを組み立て -----
    parts: list[str] = []

    # 気圧
    if req.summary.pressure is not None:
        p = req.summary.pressure
        parts.append(
            f"- 気圧: "
            f"最低 {p.min} hPa, 最高 {p.max} hPa, 変動幅 {p.delta} hPa"
        )
    else:
        parts.append("- 気圧: 情報なし")

    # 頭痛
    h = req.summary.headache
    parts.append(
        f"- 頭痛スタンプ: 合計 {h.count} 件, "
        f"最大レベル {h.max_level}, "
        f"平均レベル {h.avg_level if h.avg_level is not None else '情報なし'}"
    )

    # 睡眠
    if req.summary.sleep_hours_avg is not None:
        parts.append(f"- 平均睡眠時間: {req.summary.sleep_hours_avg} 時間")
    else:
        parts.append("- 平均睡眠時間: 情報なし")

    # 活動量
    if req.summary.activity_score_avg is not None:
        parts.append(f"- 活動量スコア平均: {req.summary.activity_score_avg}")
    else:
        parts.append("- 活動量スコア平均: 情報なし")

    stats_block = "\n".join(parts)

    user_prompt = f"""
{scope_label}（ターゲット日: {req.target_date}）の集計データは次の通りです。

{stats_block}

この情報をもとに、{scope_label}の様子を 1〜2 文の短い日本語でコメントしてください。
重くなりすぎず、やさしく振り返る感じでお願いします。
"""

    # ----- OpenAI 呼び出し -----
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # 好きなモデルに変更可
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    text = completion.choices[0].message.content.strip()

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

