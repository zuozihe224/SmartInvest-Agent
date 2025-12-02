

# SmartInvest-Agent: 基于多智能体协作的智能投研系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CrewAI](https://img.shields.io/badge/Framework-CrewAI-orange)
![LLM](https://img.shields.io/badge/Model-Qwen%202.5-green)

## 📖 项目简介

**SmartInvest-Agent** 是一个基于 **Multi-Agent（多智能体）** 架构的自动化金融分析平台。本项目旨在通过模拟真实的金融分析团队工作流，实现从数据采集、基本面分析、技术面诊断到舆情监控的全链路自动化。

系统由多个具备不同专业角色的 AI 智能体组成，它们通过协作，能够针对用户输入的股票代码（支持美股、港股及 A 股），自动生成一份逻辑严密、数据详实的**中文投资研究报告**。

本项目核心驱动模型采用 **Qwen 2.5-7B-Instruct**，结合 **yfinance** 进行实时数据获取，并使用 **Streamlit** 构建交互式前端。

## 🚀 核心功能

- **多智能体协同工作流**：
  - **数据专员 (Researcher)**：负责全球市场股票代码识别与清洗，获取基础行情数据。
  - **CFA 分析师 (Financial Analyst)**：执行深度基本面（PE/PB/EPS）与技术面（均线/趋势）双重分析，并进行风险量化评估（波动率/最大回撤）。
  - **舆情分析师 (News Analyst)**：实时抓取相关新闻，并评估市场情绪（利好/利空）。
  - **首席顾问 (Report Writer)**：汇总各方数据，输出结构化的 Markdown 投资建议书。

- **抗反爬虫机制**：内置智能限流与重试逻辑，有效规避数据源（Yahoo Finance）的频率限制。
- **全中文支持**：针对中文语境优化 Prompt，生成的报告符合国内阅读习惯。
- **可视化交互**：基于 Streamlit 的 Web 界面，操作简单直观。

## 🛠️ 技术架构

本项目主要采用了以下技术栈：

- **Orchestration Framework**: [CrewAI](https://github.com/joaomdmoura/crewAI) (负责智能体编排与任务分发)
- **LLM Core**: Qwen 2.5-7B-Instruct (通过 OpenAI 兼容接口调用)
- **Data Source**: yfinance (Yahoo Finance API)
- **UI Framework**: Streamlit
- **Tools Integration**: LangChain Tools

## 📂 项目结构

```bash
SmartInvest-Agent/
├── main.py              # 主程序入口 (Streamlit + CrewAI 逻辑)
├── requirements.txt     # 项目依赖库
├── README.md            # 项目文档
└── .env                 # (可选) 环境变量配置
```

## ⚡ 快速开始

### 1. 环境准备

确保本地已安装 Python 3.10 或以上版本。

```bash
# 克隆项目
git clone https://github.com/zuozihe224/SmartInvest-Agent.git
cd SmartInvest-Agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

本项目支持多种 OpenAI 兼容接口（如 SiliconFlow, DashScope, DeepSeek 等）。
请在 `main.py` 中配置 `API_KEY` 和 `API_BASE_URL`。

```python
# 示例配置 (main.py)
API_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
API_KEY = "sk-xxxxxxxxxxxx"
```

### 4. 运行系统

```bash
python -m streamlit run main.py
```

启动后，浏览器将自动打开 `http://localhost:8501`。


## 🔮 未来规划

- [ ] 接入 Tushare Pro 以获取更精准的 A 股数据。
- [ ] 增加 RAG (检索增强生成) 模块，支持读取本地 PDF 财报。
- [ ] 支持多只股票对比分析功能。
