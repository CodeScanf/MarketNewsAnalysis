# IntAnalysis（市场新闻智能分析系统）

一个基于 **LangGraph + FastAPI + React** 的金融新闻智能分析项目，支持：
- 新闻摄取与去重
- 实体抽取与股票影响分析
- 自然语言问答（含对话上下文）
- 知识库检索（含附件 PDF/图片）
- 每日推荐卡片
- 股票行情查询（BaoStock 本地适配）

---

## 1. 核心能力

### 1.1 新闻智能处理
- 多源新闻摄取
- 语义去重（聚类为唯一事件）
- 实体抽取（公司/行业/监管等）
- 股票影响映射（含置信度）

### 1.2 问答与知识库
- 自然语言问答（金融问答 + 通用问答路由）
- 对话短期上下文补全
- 知识库引用式问答（`/kb/query`）
- 附件临时问答（`/query-with-attachments`）

### 1.3 推荐与会话
- 基于用户历史兴趣生成推荐卡片
- 用户级会话隔离（聊天记录按账号隔离）
- 公共知识库 + 默认私有命名空间

### 1.4 股票查询（新增）
通过 BaoStock 本地适配层提供 4 个高频接口（无需独立 MCP 进程）：
- `POST /stocks/basic`：基础信息
- `POST /stocks/kdata`：K 线
- `POST /stocks/index`：指数
- `POST /stocks/valuation`：估值

统一返回：
```json
{
  "query_type": "kdata",
  "code": "sh.600000",
  "rows": [],
  "row_count": 0,
  "meta": {
    "frequency": "d",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "adjustflag": "3"
  }
}
```

---

## 2. 技术栈

### 后端
- Python 3.9+
- FastAPI
- LangGraph / LangChain
- FAISS + BM25
- spaCy
- SQLite（应用数据）
- BaoStock（股票行情）

### 前端
- React
- Vite
- Axios

---

## 3. 项目结构

```text
MarketNewsAnalysis/
├── api/                        # FastAPI 入口
│   └── main.py
├── intanalysis/                # 核心业务模块
│   ├── core.py                 # 系统主入口
│   ├── workflow.py             # LangGraph 流程
│   ├── agents.py               # 多智能体逻辑
│   ├── knowledge_base.py       # 知识库检索
│   ├── app_services.py         # 服务注入/鉴权/会话
│   └── stocks_service.py       # 股票查询服务（BaoStock 适配）
├── frontend/                   # React 前端
│   ├── src/App.jsx
│   └── src/api.js
├── tests/                      # 测试
├── dataset/                    # 本地数据目录
├── 启动文档.md                 # 快速启动说明（中文）
└── README.md
```

---

## 4. 本地安装

### 4.1 创建环境并安装依赖

```bash
# 进入项目
cd MarketNewsAnalysis

# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装项目依赖
pip install -e .

# 股票模块依赖
pip install baostock

# NLP 模型
python -m spacy download en_core_web_sm
```

也可使用仓库脚本创建 conda 环境：

```bash
bash scripts/setup_conda_env.sh
```

---

## 5. 启动方式

### 5.1 启动后端

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

可访问：
- Swagger 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

### 5.2 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认地址：`http://localhost:5173`

---

## 6. 环境变量（可选）

在项目根目录创建 `.env`：

```bash
API_KEY=your_api_key
BASE_URL=https://api.deepseek.com
MODEL_ID=deepseek-chat
```

说明：
- 若未配置可用 LLM，部分生成式能力会降级或不可用。
- 股票查询接口不依赖上述 LLM 配置。

---

## 7. API 概览

### 7.1 鉴权
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`

### 7.2 新闻与问答
- `POST /ingest`
- `POST /query`
- `POST /query-with-attachments`
- `GET /recommendations`
- `GET /stats`

### 7.3 知识库
- `POST /kb/query`
- `GET /kb/documents`
- `GET /kb/documents/{id}`
- `GET /kb/documents/{id}/file`
- `POST /kb/documents/upload`
- `POST /kb/rebuild-from-public-news`
- `GET /kb/stats`

### 7.4 股票模块
- `POST /stocks/basic`
- `POST /stocks/kdata`
- `POST /stocks/index`
- `POST /stocks/valuation`

错误语义：
- 参数错误：`400`
- 未登录：`401`
- 上游数据源异常（BaoStock）：`502`
- 其他异常：`500`

---

## 8. 股票接口示例

### 8.1 基础信息

```bash
curl -X POST "http://localhost:8000/stocks/basic" \
  -H "Content-Type: application/json" \
  -d '{"code":"600000"}'
```

### 8.2 K 线

```bash
curl -X POST "http://localhost:8000/stocks/kdata" \
  -H "Content-Type: application/json" \
  -d '{
    "code":"600000",
    "start_date":"2024-01-01",
    "end_date":"2024-01-31",
    "frequency":"d",
    "adjustflag":"3"
  }'
```

### 8.3 指数

```bash
curl -X POST "http://localhost:8000/stocks/index" \
  -H "Content-Type: application/json" \
  -d '{
    "code":"000300",
    "start_date":"2024-01-01",
    "end_date":"2024-03-01",
    "frequency":"d"
  }'
```

### 8.4 估值

```bash
curl -X POST "http://localhost:8000/stocks/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "code":"600000",
    "start_date":"2024-01-01",
    "end_date":"2024-03-01",
    "frequency":"d"
  }'
```

---

## 9. 测试

建议先跑关键回归：

```bash
pytest -q tests/test_stocks_service.py tests/test_stocks_api.py tests/test_auth_api.py
```

如需更完整测试：

```bash
pytest -q
```

前端构建验证：

```bash
npm --prefix frontend run build
```

---

## 10. 常见问题

### 10.1 401 未授权
未登录或会话失效，重新登录即可。

### 10.2 502（股票接口）
通常是 BaoStock 上游异常或依赖不可用：
- 确认已安装 `baostock`
- 确认运行环境网络可访问 BaoStock

### 10.3 前端无数据
确认后端已启动在 `8000` 端口，且前端请求代理配置正常。

---

## 11. 说明

- 启动速查请见：[启动文档.md](./启动文档.md)
- 股票接口详细对接文档请见：`.codex-api-spec/` 目录
