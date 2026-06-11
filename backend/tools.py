"""Fund Coach 数据查询工具层。

数据源优先级：
1. 天天基金 pingzhongdata JS（最快、最稳定）
2. AKShare fund_open_fund_info_em（兜底）

提供基金净值查询等实时数据获取函数，供 main.py 的 Function Calling 调用。
"""
import json
import re
from datetime import date, datetime

import akshare as ak
import pandas as pd
import requests
from datetime import date


def get_fund_nav(fund_code: str) -> dict:
    """获取某只基金的最新净值和近期收益表现。

    数据源优先级：天天基金 JS → AKShare → 报错。

    Args:
        fund_code: 基金代码，如 "005827"

    Returns:
        符合 基金架构.md 第七章出参定义的字典
    """
    errors = []

    # ---- 1. 天天基金 JS 数据源（首选） ----
    df = _fetch_from_js(fund_code)
    if df is not None and not df.empty:
        # 记录所使用的数据源（不返回给用户，仅用于调试）
        pass

    # ---- 2. AKShare 兜底 ----
    if df is None or df.empty:
        try:
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        except Exception as e:
            errors.append(f"fund_open_fund_info_em: {e}")

    # ---- 两个数据源都失败 ----
    if df is None or df.empty:
        error_msg = "数据暂不可用（" + "; ".join(errors) + "）" if errors else "数据暂不可用"
        return {"fund_code": fund_code, "error": error_msg}

    try:
        # AKShare 返回的数据统一列名
        if "净值日期" in df.columns:
            df["净值日期"] = pd.to_datetime(df["净值日期"])
        # 如果是 JS 数据源，日期已经是 Timestamp

        df = df.sort_values("净值日期").reset_index(drop=True)

        latest = df.iloc[-1]
        fund_name = str(latest.get("基金简称", ""))
        latest_nav = float(latest["单位净值"])
        nav_date = _to_date_str(latest["净值日期"])
        nav_change_pct = float(latest.get("日增长率", 0))

        today = df["净值日期"].max()

        # 近一月
        one_month_start = today - pd.DateOffset(months=1)
        one_month_data = df[df["净值日期"] >= one_month_start]
        one_month_return = _calc_return(one_month_data)

        # 近三月
        three_month_start = today - pd.DateOffset(months=3)
        three_month_data = df[df["净值日期"] >= three_month_start]
        three_month_return = _calc_return(three_month_data)

        # 年初至今
        year_start = pd.Timestamp(date(today.year, 1, 1))
        ytd_data = df[df["净值日期"] >= year_start]
        ytd_return = _calc_return(ytd_data)

        return {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "latest_nav": round(latest_nav, 4),
            "nav_date": nav_date,
            "nav_change_pct": round(nav_change_pct, 2),
            "one_month_return": one_month_return,
            "three_month_return": three_month_return,
            "ytd_return": ytd_return,
            "error": None,
        }
    except Exception as e:
        return {"fund_code": fund_code, "error": f"数据暂不可用（{e}）"}


def _fetch_from_js(fund_code: str) -> pd.DataFrame | None:
    """从天天基金 pingzhongdata JS 文件解析净值数据。

    URL 格式: http://fund.eastmoney.com/pingzhongdata/{fund_code}.js
    JS 变量: Data_netWorthTrend（每日净值数组）, fund_name（基金名称）
    """
    url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    try:
        resp = requests.get(url, timeout=5)
        resp.encoding = "utf-8"
        text = resp.text
    except Exception:
        return None

    try:
        # 提取基金名称（JS 变量名：fS_name）
        name_match = re.search(r'var fS_name\s*=\s*"([^"]+)"', text)
        fund_name = name_match.group(1) if name_match else ""

        # 提取 Data_netWorthTrend 数组
        arr_match = re.search(r'Data_netWorthTrend\s*=\s*(\[[\s\S]*?\]);', text)
        if not arr_match:
            return None

        arr_text = arr_match.group(1)

        # JS 对象键名加双引号 → 合法 JSON
        arr_text = re.sub(r'(?<!["\w])(\b\w+\b)\s*:', r'"\1":', arr_text)
        # 处理 null → None（json.loads 自动转 None）
        data = json.loads(arr_text)

        records = []
        for item in data:
            ts = item.get("x")       # 毫秒时间戳
            nav = item.get("y")      # 单位净值
            pct = item.get("equityReturn")  # 日涨跌幅，可能为 None
            if ts is None or nav is None:
                continue
            records.append({
                "净值日期": pd.Timestamp(ts, unit="ms"),
                "单位净值": float(nav),
                "日增长率": float(pct) if pct is not None else 0.0,
                "基金简称": fund_name,
            })

        if not records:
            return None

        return pd.DataFrame(records)

    except Exception:
        return None


def _to_date_str(val) -> str:
    """统一将日期值转为 'YYYY-MM-DD' 字符串。"""
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)


def _calc_return(data: pd.DataFrame) -> float | None:
    """根据净值序列计算区间收益率（百分比）。"""
    if len(data) < 2:
        return None
    first_nav = float(data.iloc[0]["单位净值"])
    last_nav = float(data.iloc[-1]["单位净值"])
    if first_nav <= 0:
        return None
    return round((last_nav / first_nav - 1) * 100, 2)


# ---------------------------------------------------------------------------
# 指数估值查询
# ---------------------------------------------------------------------------

# 指数代码映射
INDEX_MAP = {
    "000300": {"name": "沪深300", "lg_name": "沪深300"},
    "000905": {"name": "中证500", "lg_name": "中证500"},
    "000510": {"name": "中证A500", "lg_name": None},
    "399006": {"name": "创业板指", "lg_name": None},
    "000688": {"name": "科创50", "lg_name": None},
    "000001": {"name": "上证指数", "lg_name": None},
}


def get_index_valuation(index_code: str) -> dict:
    """获取指定指数的估值数据（PE/PB/百分位）。

    数据源优先级：
    1. 乐咕乐股（stock_index_pe_lg + stock_index_pb_lg）— 全历史数据，可计算百分位
    2. 中证指数官网（stock_zh_index_value_csindex）— 最近 20 期 PE 数据

    Args:
        index_code: 指数代码，如 "000300"

    Returns:
        dict 含 index_code, index_name, pe, pb, pe_percentile, pb_percentile,
        pe_50th, data_date, error
    """
    meta = INDEX_MAP.get(index_code)
    if not meta:
        return {"index_code": index_code, "error": f"不支持的指数代码: {index_code}"}

    index_name = meta["name"]
    lg_name = meta.get("lg_name")

    # ---- 优先使用乐咕乐股（全历史，可算百分位）----
    if lg_name:
        try:
            df_pe = ak.stock_index_pe_lg(symbol=lg_name)
            df_pb = ak.stock_index_pb_lg(symbol=lg_name)

            if df_pe is not None and not df_pe.empty:
                df_pe = df_pe.sort_values("日期").reset_index(drop=True)
                latest = df_pe.iloc[-1]
                data_date = str(latest["日期"])
                pe = float(latest["滚动市盈率"])
                pe_series = df_pe["滚动市盈率"].dropna().astype(float)
                pe_percentile = _calc_percentile(pe_series, pe)
                pe_50th = round(float(pe_series.median()), 2)

                pb = None
                pb_percentile = None
                if df_pb is not None and not df_pb.empty:
                    df_pb = df_pb.sort_values("日期").reset_index(drop=True)
                    pb = float(df_pb.iloc[-1]["市净率"])
                    pb_series = df_pb["市净率"].dropna().astype(float)
                    pb_percentile = _calc_percentile(pb_series, pb)

                return {
                    "index_code": index_code,
                    "index_name": index_name,
                    "pe": pe,
                    "pb": pb,
                    "pe_percentile": pe_percentile,
                    "pb_percentile": pb_percentile,
                    "pe_50th": pe_50th,
                    "data_date": data_date,
                    "error": None,
                }
        except Exception:
            # 乐咕乐股失败，降级到中证指数
            pass

    # ---- 兜底：中证指数官网（仅 PE，无 PB 百分位）----
    try:
        df = ak.stock_zh_index_value_csindex(symbol=index_code)
        if df is not None and not df.empty:
            df = df.sort_values("日期").reset_index(drop=True)
            latest = df.iloc[-1]
            pe = float(latest["市盈率1"])
            pe_series = df["市盈率1"].dropna().astype(float)
            pe_percentile = _calc_percentile(pe_series, pe)
            pe_50th = round(float(pe_series.median()), 2)
            data_date = str(latest["日期"])
            index_name = str(latest.get("指数中文简称", index_name))

            return {
                "index_code": index_code,
                "index_name": index_name,
                "pe": pe,
                "pb": None,
                "pe_percentile": pe_percentile,
                "pb_percentile": None,
                "pe_50th": pe_50th,
                "data_date": data_date,
                "error": None,
            }
    except Exception:
        pass

    # ---- 全部失败 ----
    return {"index_code": index_code, "error": "数据暂不可用"}


def _calc_percentile(series: pd.Series, current_value: float) -> float | None:
    """计算当前值在历史序列中的百分位（0~100）。"""
    series = series.dropna()
    if len(series) < 2:
        return None
    count_below = int((series <= current_value).sum())
    return round(count_below / len(series) * 100, 1)


# ---------------------------------------------------------------------------
# 宏观经济指标查询
# ---------------------------------------------------------------------------

def get_macro_data(indicator: str) -> dict:
    """获取宏观经济指标数据。

    Args:
        indicator: 指标名称 — "pmi" / "cpi" / "shibor"

    Returns:
        统一格式的字典
    """
    if indicator == "pmi":
        return _macro_pmi()
    elif indicator == "cpi":
        return _macro_cpi()
    elif indicator == "shibor":
        return _macro_shibor()
    else:
        return {"indicator": indicator, "error": f"不支持的指标: {indicator}"}


def _macro_pmi() -> dict:
    try:
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            return {"indicator": "pmi", "error": "数据暂不可用"}
        df = df.sort_values("月份").reset_index(drop=True)
        latest = df.iloc[-1]
        month_raw = str(latest["月份"])  # e.g. "2026年05月份"
        latest_month = month_raw.replace("年", "-").replace("月份", "").replace("月", "")
        latest_value = float(latest["制造业-指数"])

        # 近 6 个月序列
        recent = df.tail(6)
        data = []
        for _, row in recent.iterrows():
            m = str(row["月份"]).replace("年", "-").replace("月份", "").replace("月", "")
            data.append({"month": m, "value": float(row["制造业-指数"])})

        # 趋势判断
        vals = [float(v) for v in recent["制造业-指数"].tail(3)]
        if all(v > 50 for v in vals):
            trend = f"连续{sum(1 for v in vals if v > 50)}个月处于扩张区间"
        elif all(v < 50 for v in vals):
            trend = f"连续{sum(1 for v in vals if v < 50)}个月处于收缩区间"
        else:
            trend = "近期处于波动状态"

        return {
            "indicator": "pmi",
            "latest_value": latest_value,
            "latest_month": latest_month,
            "trend": trend,
            "data": data,
            "error": None,
        }
    except Exception:
        return {"indicator": "pmi", "error": "数据暂不可用"}


def _macro_cpi() -> dict:
    try:
        df = ak.macro_china_cpi()
        if df is None or df.empty:
            return {"indicator": "cpi", "error": "数据暂不可用"}
        df = df.sort_values("月份").reset_index(drop=True)
        latest = df.iloc[-1]
        month_raw = str(latest["月份"])
        latest_month = month_raw.replace("年", "-").replace("月份", "").replace("月", "")
        latest_value = float(latest["全国-当月"])
        yoy = float(latest["全国-同比增长"])

        recent = df.tail(6)
        data = []
        for _, row in recent.iterrows():
            m = str(row["月份"]).replace("年", "-").replace("月份", "").replace("月", "")
            data.append({"month": m, "value": float(row["全国-当月"]), "yoy": float(row["全国-同比增长"])})

        return {
            "indicator": "cpi",
            "latest_value": latest_value,
            "latest_month": latest_month,
            "yoy": yoy,
            "data": data,
            "error": None,
        }
    except Exception:
        return {"indicator": "cpi", "error": "数据暂不可用"}


def _macro_shibor() -> dict:
    try:
        df = ak.rate_interbank(
            market="上海银行同业拆借市场",
            symbol="Shibor人民币",
            indicator="隔夜",
        )
        if df is None or df.empty:
            return {"indicator": "shibor", "error": "数据暂不可用"}
        df = df.sort_values("报告日").reset_index(drop=True)
        latest = df.iloc[-1]
        latest_value = float(latest["利率"])
        latest_date = str(latest["报告日"])

        recent = df.tail(6)
        data = []
        for _, row in recent.iterrows():
            data.append({"month": str(row["报告日"]), "value": float(row["利率"])})

        return {
            "indicator": "shibor",
            "latest_value": latest_value,
            "latest_month": latest_date,
            "data": data,
            "error": None,
        }
    except Exception:
        return {"indicator": "shibor", "error": "数据暂不可用"}
