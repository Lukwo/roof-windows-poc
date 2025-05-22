# app.py  –  Roof-Window Assistant with fuzzy-SQL and clarifications
# ──────────────────────────────────────────────────────────────────
import os, io, re, json, duckdb, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
import openai
from rapidfuzz import process, fuzz          # NEW

load_dotenv()
st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───── sidebar ────────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)

st.sidebar.markdown("### Roof-Window Knowledge-Bot\n_UK market – PoC_")

SAMPLES = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide",
]
with st.sidebar.expander("ℹ️ Try one (click)"):
    for q in SAMPLES:
        st.button(q, on_click=lambda x=q: st.session_state.update(prompt=x))

if st.sidebar.button("🔄 Reset chat"):
    st.session_state.pop("chat", None)
    st.experimental_rerun()

# ───── data ───────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/roof_windows_uk.parquet")

roof = load_data()
COLS = list(roof.columns)

# ───── user prompt ────────────────────────────────────────────────
prompt = st.text_input("Ask a question about UK roof windows:", key="prompt")
if not prompt:
    st.stop()

# ───── chat history / system prompt ───────────────────────────────
SYSTEM = f"""
You are a strict data assistant for UK roof-window data.

Allowed columns:
    {', '.join(sorted(COLS))}

Return ONLY a function call with JSON:
  sql   – a SELECT that uses ONLY those columns and table 'roof'.
  excel – true if user explicitly asks for a downloadable Excel.

Never invent new column names. If the user typo looks ambiguous
(e.g. 'glas type'), guess the closest valid column.
"""

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "system", "content": SYSTEM}]
st.session_state.chat.append({"role": "user", "content": prompt})

# ───── OpenAI call ────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",                 # or gpt-3.5-turbo-0125
        messages=st.session_state.chat,
        tools=[{
            "type": "function",
            "function": {
                "name": "execute_query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql":   {"type": "string"},
                        "excel": {"type": "boolean"},
                    },
                    "required": ["sql", "excel"],
                },
            },
        }],
        tool_choice="auto",
    )
except RateLimitError:
    st.error("🛑 OpenAI quota exhausted – top-up billing or swap keys, then Rerun.")
    st.stop()

st.session_state.chat.append(resp.choices[0].message)

tool_call = resp.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
sql = args["sql"]
need_excel = args["excel"]

# ───── Fuzzy column mapping  NEW ──────────────────────────────────
bad_cols = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql)
replacements = {}
for token in set(bad_cols):
    if token.lower() in COLS:
        continue
    match, score, _ = process.extractOne(
        token, COLS, scorer=fuzz.WRatio, score_cutoff=70
    )
    if match:
        replacements[token] = match
    else:
        st.warning(
            f"Unsure what you meant by **{token}**. "
            "Please rephrase or choose one of: "
            f"{', '.join(sorted(COLS)[:20])} …"
        )
        st.stop()

for wrong, right in replacements.items():
    sql = re.sub(rf"\b{wrong}\b", right, sql)

st.code(sql, language="sql")

# ───── execute SQL ────────────────────────────────────────────────
try:
    result = duckdb.query_df(roof, "roof", sql).df()
except Exception as e:
    st.error(f"⛔ SQL error: {e}")
    st.stop()

if result.empty:
    st.warning("No rows matched that query.")
    st.stop()

# ───── table with highlight ───────────────────────────────────────
def hl(row):
    return [
        "background-color:#d2ead2" if c.endswith("_num") and pd.notna(row[c]) and row[c] <= 1
        else "" for c in row.index
    ]

st.dataframe(result.style.apply(hl, axis=1), use_container_width=True)

# ───── optional Excel download ────────────────────────────────────
if need_excel:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        result.to_excel(xl, sheet_name="RoofWindows", index=False)
    st.download_button(
        "⬇️ Download filtered Excel",
        data=buf.getvalue(),
        file_name="roof_windows_uk.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
