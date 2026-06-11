"""Fund Coach 后端服务。

提供 /api/chat 接口，集成 DeepSeek Chat + Function Calling（基金净值查询）。
"""
import json
import os
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from tools import get_fund_nav, get_index_valuation, get_macro_data

load_dotenv()

app = FastAPI(title="Fund Coach API")

# ---------------------------------------------------------------------------
# CORS（允许前端开发时跨域）
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 静态文件托管（前端页面）
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# ---------------------------------------------------------------------------
# DeepSeek 客户端
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ---------------------------------------------------------------------------
# 请求 / 响应 模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    holdings: list
    user_profile: dict = {}
    conversation_history: list = []
    new_message: str
    mode: str = "chat"


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个专业的基金定投助手（Fund Coach），基于**核心-卫星策略**帮助用户分析基金投资组合。

【核心-卫星框架】
- 核心仓（建议 70-80%）：以宽基指数基金为主体，长期持有
  - 标准宽基：沪深300、中证500、中证A500、纳斯达克100、标普500 等
  - 弹性定义：用户长期看好并持续投入的主线指数（如科创创业50）也可归入核心仓
  - 核心仓的关键判断标准：覆盖多个行业 / 计划持有 3 年以上 / 持续定投
- 卫星仓（建议 20-30%）：行业/主题基金，小仓试错，培养前瞻判断力
- 用户持仓中包含什么就分析什么，不预设用户应该买什么
- 尊重用户对持仓的分类，在用户分类基础上优化配比，而非强行重新归类
- **用户持仓信息中已包含基金名称等字段，分析时以用户提供的信息为准，不要自行根据基金代码推测**

【分析逻辑】
分析用户持仓时，按以下顺序：
1. 理解用户的持仓结构——哪些是核心仓、哪些是卫星仓（可以问用户或基于基金性质判断）
2. 评估核心仓配比是否合理：各宽基之间的分散度、市场覆盖（A股/港股/美股）、估值水平
3. 评估卫星仓方向是否符合产业趋势：用户持有的行业是否有中长期逻辑支撑
4. 给出结论：结构合理则建议维持，结构偏差则给出调整方向

【回答风格 — 核心原则】
- **匹配用户语气**：用户问得短，你回得短。用户说"在吗"，你回"在的，想聊什么"而不是直接输出整篇分析。用户问具体问题，给具体答案。
- **直接给结论**：不列 ABC 选项让用户自己选。直接说"建议做 X，原因是 Y"。用户追问再展开备选。
- **克制信息密度**：除非用户明确要求"详细分析"，否则只说核心判断。不把数据表、估值分位、收益率一次性全堆出来。
- **拒绝谄媚和套话**：不说"在这个信息爆炸的时代""值得注意的是""不妨试试""一起探索""重要性不言而喻"这类空话。
- **自然对话感**：句子长短错落，不排比堆砌。不说"请务必注意"这种空洞警示词。

【引导式分析框架】
你的角色是分析引导员，不是投资顾问。每次回答按以下结构：

**1. 诊断结论（必须有）**
给出明确的判断，不模棱两可。例如"你的组合集中度偏高"而不是"你的组合可能有点集中"。但不输出具体操作指令（不说不买/不说不卖）。

**2. 关注引导（核心价值）**
告诉用户应该关注什么信息、什么时间节点、什么事件。例如"关注月底公布的PCE数据""关注下季度美联储利率决议"。可推荐外部信息源，不编造具体数据。

**3. 回访入口（可选）**
留下一个开放性问题或建议，让用户可以回来继续聊。例如"等数据出来后再来分析要不要调整"。

**4. 首次对话提示**
如果用户是第一次进入对话（无对话历史），且用户画像为空（只有默认值），在首条回复末尾主动问一句："你还没做过投资评测，可以去投资画像页做一下，这样我的分析会更准。"

【投资评测引导 — 自适应分支】
你的目标是通过自然对话了解用户的投资水平，而不是让用户填表单。

第一问决定分支：
首轮问一个开放式问题来判断用户水平，例如"你之前接触过基金投资吗？平时会关注市场动态吗？"
根据回答识别用户属于哪个分支，后续对话自然适配：

| 用户特征 | 分支 | 语言风格 |
|---------|------|---------|
| "没接触过"、"刚了解"、"不太懂" | 刚接触 | 不用术语。"跌了"不说"回撤"，"贵不贵"不说"估值分位"。用生活类比解释概念。 |
| "买过一些"、"定投了 X 年"、"懂一点" | 有经验 | 可以正常使用术语，但不要一次性堆砌。先给结论，用户追问再展开。 |
| "投资 X 年了"、"做过资产配置"、"看宏观" | 老手 | 直接聊策略、结构、时局。可以讨论卫星仓方向、再平衡周期、估值水位。 |

过渡原则：
- 自然过渡，不使用"由于你是新手…"、"根据你的经验水平…"这类生硬衔接语
- 用内容本身适配用户水平，而不是贴标签
- 如果用户表现出超越当前分支的知识，自动升档

【评测规则 — 强制】
你处于评测模式时（从你主动发起评测到调用 update_user_profile 之间），遵守以下规则：

1. ⚠️ **强制调用**：当你收集到至少 3 个维度的明确信息（经验/风险偏好/目标/持仓行为），且自己能判断画像时，必须立即调用 update_user_profile。调用时机在"给用户回复之前"。不要提前告知用户你要调用——直接调。
   正确：用户说完→你调用 FC→然后回复"好了，已更新你的画像..."
   错误：用户说完→你说"差不多了，我先记录一下"→调用 FC→回复（晚了）

2. ⚠️ **禁止元评论**：不可以说"再问一个就够"、"差不多了"、"根据你的回答"这类关于评测本身的话。直接问下一题或直接调用 FC。

3. ⚠️ **评测期间拒答非评测问题**：评测过程中，用户问持仓分析、查估值、查净值等非评测问题，统一回复"请先完成评测，之后我会帮你分析"。评测完成（调用 FC 后）才进入正常分析模式。

**置信度赋值参考**：1-2 个维度 0.3-0.5（不要调用），3 个维度且有明确信号 0.6-0.8，4-5 个维度且信号一致 0.8-1.0。不要求一次性完成，后续对话可补充。
动态校正：每次对话可补充画像。已有画像且新信息与旧矛盾时需更充分证据。已确认信息不重复问。多次对话信号一致→画像稳定→调整频率自然降低。

【内容约束】
- 不主动推荐具体基金（包括行业基金），不含买卖时点建议
- 如果用户持有行业基金，帮助分析其趋势逻辑；如果没持有，不主动推荐
- 回答时若涉及调整建议，必须附带逻辑依据，让用户理解"为什么"
- **展示持仓数据时，必须先列出每只基金的原始定投金额（如"每周20元"）再展示占比/折算等衍生数据。用户无法直接从占比反推出原始输入，所以两者缺一不可。**
- **禁止使用"如果是A...如果是B..."的条件选择句式**，这会把判断责任推回给用户。如果确实存在不确定性，直接说"目前信号不明确"并告诉用户需要补充什么信息，而不是给两个选项让他自己选。
- 不知道就明确说不知道，不编造

【时局分析准则】
只有在出现**确定性较高的中长期趋势信号**时，才给出调整建议。以下算明确信号：
- 美联储等主要央行的利率周期明确转向（非单次数据波动）
- 国内明确的政策转向或产业支持政策密集出台
- 主要指数估值处于历史极端分位且基本面有配合
- 地缘/宏观事件出现结构性变化（非短期冲击）

如果没有上述级别的信号，应明确回答：
"当前各赛道趋势不明朗，建议维持现有定投计划。"
调整建议必须附带时局依据（关键数据、政策原文、逻辑链），让用户理解"为什么这次调整值得做"。

【严格模式】
- 当通过工具函数获取数据时，如果返回的数据中包含 error 字段，明确告知用户数据暂不可用
- 不基于大模型自身知识推测实时净值数据
- ⚠️ 如果通过工具获取数据返回了 error，直接告知用户数据暂不可用即可。严禁根据基金代码推测基金名称、类型、投向等任何信息。例如：用户查询 013304 失败时，不能说"013304 可能是某只纳斯达克基金"，而是只说"013304 数据暂不可用"。

【估值参考】
pe_percentile < 20% 为低估区间，20-60% 正常，> 60% 高估。只说估值状态，不给买卖建议。"""


# ---------------------------------------------------------------------------
# Function Calling 定义
# ---------------------------------------------------------------------------
FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_fund_nav",
            "description": "获取某只基金的最新净值和近期收益表现",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_code": {
                        "type": "string",
                        "description": "基金代码，如 005827",
                    }
                },
                "required": ["fund_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_valuation",
            "description": "获取指定指数的估值数据（PE/PB/百分位），用于判断宽基指数估值贵不贵。",
            "parameters": {
                "type": "object",
                "properties": {
                    "index_code": {
                        "type": "string",
                        "enum": ["000300", "000905", "000510", "399006", "000688", "000001"],
                        "description": "指数代码：000300=沪深300, 000905=中证500, 000510=中证A500, 399006=创业板指, 000688=科创50, 000001=上证指数",
                    },
                    "index_name": {
                        "type": "string",
                        "description": "指数名称（非必填，仅用于可读性）",
                    },
                },
                "required": ["index_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_data",
            "description": "获取宏观经济指标数据（PMI/CPI/Shibor），用于判断当前经济周期和货币政策走向。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["pmi", "cpi", "shibor"],
                        "description": "指标名称：pmi=采购经理人指数, cpi=居民消费价格指数, shibor=上海银行间同业拆放利率",
                    },
                },
                "required": ["indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "根据对话中收集到的用户信息，更新用户投资画像。收集足够信号（3-5个维度）后再调用，不要过早调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "enum": ["保守型", "稳健型", "平衡型", "进取型", "激进型"],
                        "description": "用户风险承受等级",
                    },
                    "investment_experience": {
                        "type": "string",
                        "enum": ["刚接触", "1年以内", "1-3年", "3-5年", "5年以上"],
                        "description": "用户投资经验",
                    },
                    "investment_goal": {
                        "type": "string",
                        "enum": ["资金保值", "长期增值", "收益最大化"],
                        "description": "用户投资目标",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "AI 对评测结果的置信度，0.0~1.0",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["risk_level", "confidence"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/frontend/index.html#/")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """对话接口：接收用户输入 + 持仓/画像 + 对话历史，返回 AI 回答。"""
    try:
        pending_profile_update = None

        if req.mode == "evaluate":
            EVAL_PROMPT = """你是一个投资评测助手。你的唯一任务是通过 3 个问题完成用户画像评测。

**【评测流程 — 严格模板】**
一共 3 个问题，所有用户都走这 3 题，但第 2 题根据第 1 题的答案**自适应语言风格**。

**Q1（必问）**："你投资基金多久了？你理解定投的理念吗？"
→ 根据回答识别分支：
| 关键词 | 分支 | 后续语言风格 |
|--------|------|------------|
| "没接触过"、"刚了解"、"不太懂" | 刚接触 | 不用术语。"跌了"不说"回撤"，用生活类比 |
| "买过一些"、"定投了 X 年"、"懂一点" | 有经验 | 正常用术语，不堆砌 |
| "投资 X 年了"、"做过资产配置"、"看宏观" | 老手 | 聊策略和结构 |

**Q2（必问，按分支适配语言）**：
- 刚接触 → "你觉得选对东西重要吗？如果买的东西一直跌，你一般怎么办？打算拿多久？"
- 有经验 → "你认为好的标的重要吗？如果你认为好的标的持续下跌，一般怎么应对？另外，你打算持有你的标的多久？"
- 老手 → "你选标的看重什么？如果持仓持续跌，你怎么判断是加仓还是走？一般持有多久？"

**Q3（必问）**："你定投的主要目标是什么？——保值、长期增值、还是收益最大化？"

⚠️ **用户答完 Q3 后必须立即调用 update_user_profile，禁止追加任何问题。**
⚠️ **评测期间不分析持仓、不查净值、不查估值。**
⚠️ **禁止元评论**：不可以说"再问一个就够"、"差不多了"这类话。"""
            messages = [{"role": "system", "content": EVAL_PROMPT}]
            eval_tools = [FUNCTIONS[3]]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            eval_tools = FUNCTIONS
            if req.user_profile:
                profile_note = f"用户画像：{json.dumps(req.user_profile, ensure_ascii=False)}"
                # 检查是否有真实画像字段（非默认空值）
                has_real_profile = bool(req.user_profile.get("risk_level") and
                    req.user_profile.get("risk_level") != "未评测")
                if not has_real_profile:
                    profile_note += "\n注意：用户尚未做过投资评测，默认按稳健型处理。请在回复末尾主动询问用户是否要做评测。"
                messages.append({
                    "role": "system",
                    "content": profile_note,
                })

            if req.holdings:
                messages.append({
                    "role": "system",
                    "content": f"用户持仓：{json.dumps(req.holdings, ensure_ascii=False)}",
                })

        for msg in req.conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": req.new_message})

        # ---- 短消息守卫：用户没说正事，不回正事 ----
        # 只在对话开头（无历史）时触发，持续对话中的短指代不过滤
        if len(req.conversation_history) == 0:
            _SHORT_MSG_KEYWORDS = ["基金", "持仓", "净值", "分析", "时局", "定投", "调整",
                "推荐", "查", "收益", "涨", "跌", "趋势", "配置", "比例", "风险",
                "卫星", "核心", "行业", "指数", "宽基", "赛道", "回撤", "评估",
                "沪深300", "中证", "纳斯达克", "双创", "科创", "评测", "开始评测"]
            _msg = req.new_message.strip()
            if len(_msg) <= 10 and not any(kw in _msg for kw in _SHORT_MSG_KEYWORDS):
                _replies = [
                    "在的，想聊什么？",
                    "在呢，有持仓想看看吗？",
                    "你说。",
                    "嗯，我在。"
                ]
                import random
                return {"reply": random.choice(_replies)}

        # 第一轮：调用 DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=eval_tools,
            tool_choice="auto",
        )

        choice = response.choices[0].message

        # 如果返回了 function_call，执行工具并再次调用
        if choice.tool_calls:
            messages.append(choice)

            for tool_call in choice.tool_calls:
                if tool_call.function.name == "get_fund_nav":
                    args = json.loads(tool_call.function.arguments)
                    result = get_fund_nav(args.get("fund_code", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                elif tool_call.function.name == "get_index_valuation":
                    args = json.loads(tool_call.function.arguments)
                    result = get_index_valuation(args.get("index_code", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                elif tool_call.function.name == "get_macro_data":
                    args = json.loads(tool_call.function.arguments)
                    result = get_macro_data(args.get("indicator", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                elif tool_call.function.name == "update_user_profile":
                    args = json.loads(tool_call.function.arguments)
                    confidence = args.get("confidence", 0)
                    new_risk_level = args.get("risk_level")

                    # ---- 阻尼机制：已有画像时覆盖 risk_level 需更高阈值 ----
                    old_profile = req.user_profile or {}
                    old_risk_level = old_profile.get("risk_level")
                    old_confidence = old_profile.get("confidence", 0)
                    old_updated_at = old_profile.get("updated_at")

                    can_update = False
                    if old_risk_level and new_risk_level and new_risk_level != old_risk_level:
                        # 需要覆盖已有画像的 risk_level —— 启用阻尼
                        days_passed = 999  # 无 updated_at 视为可覆盖
                        if old_updated_at:
                            try:
                                dt = datetime.fromisoformat(old_updated_at.replace("Z", "+00:00"))
                                days_passed = (datetime.now(timezone.utc) - dt).days
                            except (ValueError, TypeError):
                                days_passed = 999
                        can_update = confidence >= 0.85 and (old_confidence < 0.5 or days_passed >= 7)
                        if not can_update:
                            msg = "信号不足，暂不更新风险等级。当前画像已趋于稳定，需要更多一致的对话证据才能调整。"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"success": False, "message": msg}, ensure_ascii=False),
                            })
                    else:
                        # 无旧画像 / risk_level 不变 —— 走常规规则
                        can_update = confidence >= 0.6

                    if can_update:
                        profile_update = {
                            "risk_level": new_risk_level,
                            "investment_experience": args.get("investment_experience"),
                            "investment_goal": args.get("investment_goal"),
                            "confidence": confidence,
                        }
                        pending_profile_update = profile_update
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"success": True, "message": "画像已更新"}, ensure_ascii=False),
                        })
                    elif not old_risk_level or (new_risk_level and new_risk_level == old_risk_level):
                        # 常规规则不满足（confidence < 0.6 且不是阻尼拦截的情况）
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"success": False, "message": "置信度不足，暂不更新"}, ensure_ascii=False),
                        })

            # 第二轮：带工具结果再次调用 DeepSeek
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=eval_tools,
                tool_choice="auto",
            )
            reply = response.choices[0].message.content or ""
        else:
            reply = choice.content or ""

        response_data = {"reply": reply}
        if pending_profile_update:
            response_data["profile_update"] = pending_profile_update
        return response_data

    except Exception as e:
        return {"reply": f"抱歉，服务暂时出错了，请稍后重试。（{type(e).__name__}）"}


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
