# app.py  –  Roof-Window Assistant  –  UK PoC
# ─────────────────────────────────────────────────────────────────────
import os, io, re, json, duckdb, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
import openai

load_dotenv()                                    # allows local .env testing
st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ─────────── 0 · sidebar (logo, samples, reset) ─────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)

st.sidebar.markdown(
    "### Roof-Window Knowledge-Bot  \n"
    "_UK market – Proof of Concept_"
)

sample_qs = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide"
]
with st.sidebar.expander("ℹ️ Try one (click)"):
    for q in sample_qs:
        st.button(q, on_click=lambda x=q: st.session_state.update(prompt=x))

# reset-chat button
if st.sidebar.button("🔄 Reset chat"):
    st.session_state.pop("chat", None)
    st.experimental_rerun()

# ─────────── 1 · load cleaned parquet (cached) ──────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/roof_windows_uk.parquet")

roof = load_data()

# ─────────── 2 · prompt box ─────────────────────────────────────────
prompt = st.text_input("Ask a question about UK roof windows:", key="prompt")
if not prompt:
    st.stop()

# ─────────── 3 · build / maintain chat history ──────────────────────
SYSTEM = """
You are a strict data assistant.  Answer ONLY by calling the function below.
Return JSON with keys:
  sql   – a SELECT statement that answers the question using table 'roof'
  excel – true if user wants a downloadable Excel, else false
Never add extra keys or free-form text.
"""

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "system", "content": SYSTEM}]

# add current user message
st.session_state.chat.append({"role": "user", "content": prompt})

# ─────────── 4 · call OpenAI with tool-calling ──────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",                     # switch to gpt-3.5-turbo-0125 if preferred
        messages=st.session_state.chat,
        tools=[{
            "type": "function",
            "function": {
                "name": "execute_query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql":   {"type": "string"},
                        "excel": {"type": "boolean"}
                    },
                    "required": ["sql", "excel"]
                }
            }
        }],
        tool_choice="auto",
    )
except RateLimitError:
    st.error(
        "🛑 OpenAI quota exhausted for this API key.  "
        "Top-up billing or paste a different key under **Settings → Secrets**, "
        "then click **Rerun**."
    )
    st.stop()

# store assistant reply in history for context next turn
st.session_state.chat.append(response.choices[0].message)

# ─────────── 5 · parse tool arguments (JSON string → dict) ──────────
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

sql_query   = args["sql"]
need_excel  = args["excel"]

# show SQL for debugging
st.code(sql_query, language="sql")

# ─────────── 6 · run SQL safely with DuckDB ─────────────────────────
try:
    result = duckdb.query_df(roof, "roof", sql_query).df()
except Exception as e:
    st.error(f"⛔ SQL error: {e}")
    st.stop()

# warn if no rows matched
if result.empty:
    st.warning("No rows matched that query.")
    st.stop()

# ─────────── 7 · display table with highlights ──────────────────────
def highlight(row):
    styles = []
    for c in row.index:
        if c.endswith("_num") and pd.notna(row[c]) and row[c] <= 1:
            styles.append("background-color:#d2ead2")
        else:
            styles.append("")
    return styles

st.dataframe(
    result.style.apply(highlight, axis=1),
    use_container_width=True
)

# ─────────── 8 · optional Excel download ────────────────────────────
if need_excel:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        result.to_excel(xl, sheet_name="RoofWindows", index=False)
        ws = xl.sheets["RoofWindows"]
        green = xl.book.add_format({"bg_color": "#d2ead2"})
        for col_idx, col in enumerate(result.columns):
            if col.endswith("_num"):
                ws.conditional_format(
                    1, col_idx, len(result), col_idx,
                    {"type": "cell", "criteria": "<=", "value": 1, "format": green}
                )
    st.download_button(
        "⬇️ Download filtered Excel",
        data=buf.getvalue(),
        file_name="roof_windows_uk.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ─────────────────────────────────────────────────────────────────────
# end of file
