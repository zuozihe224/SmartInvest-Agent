import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from crewai import Crew, Agent, Task, Process
from langchain.tools import tool 
from langchain_openai import ChatOpenAI 
import numpy as np
import streamlit as st

# ================== 核心配置区域 ==================

# 请填入你的配置
API_BASE_URL = "https://api.siliconflow.cn/v1" # 或者是阿里云/本地的地址
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
API_KEY = "sk-khukbdqsvdfgfyqvpgsumqvrlpsgzitemhiqczugogvnyoph" # 记得填你的 Key

# =================================================

# 初始化 LLM
llm_model = ChatOpenAI(
    model_name=MODEL_NAME, 
    openai_api_base=API_BASE_URL,
    openai_api_key=API_KEY,
    temperature=0.1,
    max_tokens=4096
)

# ================= 工具函数 (修复了 Rate Limit 问题) =================

def safe_yf_ticker(ticker):
    """一个安全的获取Ticker的辅助函数，带重试机制"""
    # 强制休息3秒，防止被 Yahoo 封锁
    time.sleep(3) 
    try:
        obj = yf.Ticker(ticker)
        # 尝试访问一下 info 触发请求，看是否报错
        _ = obj.info 
        return obj
    except Exception:
        # 如果失败，再等3秒重试一次
        time.sleep(3)
        return yf.Ticker(ticker)

@tool
def get_basic_stock_info(ticker: str) -> pd.DataFrame:
    """获取股票基本信息。参数 ticker: 股票代码 (如 AAPL, 0700.HK)"""
    try:
        stock = safe_yf_ticker(ticker)
        info = stock.info
        
        # 这里的 get 防止某些字段不存在报错
        basic_info = pd.DataFrame({
            '名称': [info.get('longName', ticker)],
            '行业': [info.get('industry', 'N/A')],
            '市值': [info.get('marketCap', 'N/A')],
            '当前价格': [info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))],
            '52周最高': [info.get('fiftyTwoWeekHigh', 'N/A')]
        })
        return basic_info
    except Exception as e:
        return pd.DataFrame({'Error': [f"数据获取失败，可能是请求太快: {str(e)}"]})

@tool
def get_fundamental_analysis(ticker: str, period: str = '1y') -> pd.DataFrame:
    """基本面分析。"""
    try:
        stock = safe_yf_ticker(ticker)
        info = stock.info
        return pd.DataFrame({
            '市盈率(PE)': [info.get('trailingPE', 'N/A')],
            '市净率(PB)': [info.get('priceToBook', 'N/A')],
            '每股收益(EPS)': [info.get('trailingEps', 'N/A')],
            '利润率': [info.get('profitMargins', 'N/A')],
            '营收增长': [info.get('revenueGrowth', 'N/A')]
        })
    except Exception:
        return pd.DataFrame({'Status': ['数据暂时无法获取，请参考技术面']})

@tool
def get_stock_risk_assessment(ticker: str, period: str = "1y") -> pd.DataFrame:
    """风险评估。"""
    try:
        stock = safe_yf_ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            return pd.DataFrame({'Error': ['无历史数据']})
            
        returns = history['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        peak = history['Close'].cummax()
        max_drawdown = ((history['Close'] - peak) / peak).min()
        
        return pd.DataFrame({'年化波动率': [volatility], '最大回撤': [max_drawdown]})
    except Exception:
        return pd.DataFrame({'Status': ['风险数据计算失败']})

@tool
def get_technical_analysis(ticker: str, period: str = "1y") -> pd.DataFrame:
    """技术面分析 (均线)。"""
    try:
        stock = safe_yf_ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            return pd.DataFrame({'Error': ['无历史数据']})

        history['SMA50'] = history['Close'].rolling(50).mean()
        latest = history.iloc[-1]
        trend = "牛市趋势" if latest['Close'] > latest['SMA50'] else "震荡/熊市"
        
        return pd.DataFrame({
            '当前价': [latest['Close']],
            '50日均线': [latest['SMA50']],
            '趋势判断': [trend]
        })
    except Exception:
        return pd.DataFrame({'Status': ['技术指标计算失败']})

@tool
def get_stock_news(ticker: str, limit: int = 3) -> pd.DataFrame:
    """获取新闻。"""
    try:
        time.sleep(2) # 获取新闻前也睡一会
        stock = yf.Ticker(ticker)
        news = stock.news[:limit]
        return pd.DataFrame([{'标题': n['title'], '时间': datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d')} for n in news])
    except Exception:
        return pd.DataFrame({'News': ['暂无最新新闻']})

# ================= Agent 定义 =================

stock_researcher = Agent(
    llm=llm_model,
    role="资深证券数据专员",
    goal="准确识别用户查询的股票代码，并提取关键市场数据。",
    backstory="你精通全球股票市场代码。如果获取数据失败，请不要由于，直接告诉下游同事数据缺失即可。",
    tools=[get_basic_stock_info],
    verbose=True,
    allow_delegation=False
)

financial_analyst = Agent(
    llm=llm_model,
    role="CFA持证分析师",
    goal="基于数据进行深度的基本面与技术面诊断。",
    backstory="你擅长分析。如果上一步的数据获取失败（显示Error），请基于你的通用知识对该公司的行业地位进行定性分析，不要强行编造数据。",
    tools=[get_technical_analysis, get_fundamental_analysis, get_stock_risk_assessment],
    verbose=True,
    allow_delegation=False
)

news_analyst = Agent(
    llm=llm_model,
    role="金融舆情分析师",
    goal="分析新闻对股价的潜在影响。",
    backstory="你能够快速阅读新闻标题。",
    tools=[get_stock_news],
    verbose=True
)

report_writer = Agent(
    llm=llm_model,
    role="首席投资顾问",
    goal="汇总分析，输出一份逻辑严密、语气专业的中文研报。",
    backstory="你的报告需要结构清晰。如果前面的数据缺失，请在报告中注明“部分实时数据暂时不可用”，并重点分析公司的商业模式。",
    tools=[],
    verbose=True,
    allow_delegation=False
)

# ================= Task 定义 =================

task1 = Task(
    description='从用户输入 "{query}" 中识别股票，获取基础数据。',
    expected_output="股票代码、名称及基础行情数据。",
    agent=stock_researcher
)

task2 = Task(
    description='对该股票进行基本面和技术面分析，评估风险。如果数据获取失败，请进行定性分析。',
    expected_output="包含估值分析、趋势判断和风险指标的详细分析段落。",
    agent=financial_analyst,
    context=[task1]
)

task3 = Task(
    description='获取最新新闻并评估市场情绪。',
    expected_output="近期新闻摘要及情绪倾向。",
    agent=news_analyst,
    context=[task1]
)

task4 = Task(
    description='''
    撰写最终投资分析报告。
    要求：
    1. 全文中文，Markdown 格式。
    2. 包含：【核心结论】、【基本面透视】、【技术面与风险】、【消息面】、【投资建议】。
    3. 如果某些数据缺失，请用通用的行业知识补充分析，保证报告完整性。
    用户输入: {query}
    ''',
    expected_output="一份完整的中文投资研报。",
    agent=report_writer,
    context=[task1, task2, task3]
)

crew = Crew(
    agents=[stock_researcher, financial_analyst, news_analyst, report_writer],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    manager_llm=llm_model
)

# ================= UI 部分 =================

st.set_page_config(page_title="Qwen 智能投研系统", layout="wide")
st.title("📈 智能投研 Agent 系统 (Powered by Qwen 2.5)")

with st.sidebar:
    st.header("控制台")
    query = st.text_area("输入分析目标", value="分析一下 腾讯控股(0700.HK) 的近期投资价值", height=100)
    run_btn = st.button("开始分析")
    st.info("当前模型: Qwen 2.5-7B-Instruct")

if run_btn:
    if "sk-" not in API_KEY and "localhost" not in API_BASE_URL:
        st.error("请在代码第17行填入正确的 API Key！")
    else:
        st.status("AI 团队正在协作分析中... (已开启限流保护模式)", expanded=True)
        try:
            result = crew.kickoff(inputs={"query": query})
            st.success("分析完成！")
            st.markdown("## 📊 最终研报")
            st.markdown(result)
        except Exception as e:
            st.error(f"运行出错: {str(e)}")