# app.py  –  Roof-Window Assistant with fuzzy fallback (safe unpack)
# ------------------------------------------------------------------
import os, io, re, json, duckdb, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
import openai
from rapidfuzz import process, fuzz           # fuzzy matcher

load_dotenv()
st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───────── sidebar ────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)

st.sidebar.markdown("### Roof-Window Knowledge-Bot\n_UK market · PoC_")

DEMO_QUESTIONS = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide",
]
with st.sidebar.expander("ℹ️ Try one (click)"):
    for q in DEMO_QUESTIONS:
        st.button(q, on_click=lambda x=q: st.session_state.update(prompt=x))

if st.sidebar.button("🔄 Reset chat"):
    st.session_state.pop("chat", None)
    st.experimental_rerun()

# ───────── data ───────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/roof_windows_uk.parquet")

roof = load_data()
COLUMNS = list(roof.columns)

# ───────── prompt ─────────────────────────────────────────────────
prompt = st.text_input("Ask a question about UK roof windows:", key="prompt")
if not prompt:
    st.stop()

# ───────── chat memory & system prompt ────────────────────────────
SYSTEM = f"""
You are a strict data-assistant for UK roof-window data.

Allowed columns:
    {', '.join(sorted(COLUMNS))}

Return ONLY a function call with JSON:
  sql   – a SELECT that uses ONLY those columns and table 'roof'.
  excel – true if the user explicitly asks for a downloadable Excel file.

Never invent new column names. If the user’s wording is ambiguous,
guess the closest valid column name.
"""

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "system", "content": SYSTEM}]
st.session_state.chat.append({"role": "user", "content": prompt})

# ───────── OpenAI call ────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",                      # or gpt-3.5-turbo-0125
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
    st.error("🛑 OpenAI quota exhausted. Top-up or swap keys, then **Rerun**.")
    st.stop()

st.session_state.chat.append(resp.choices[0].message)

tool_call = resp.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
sql = args["sql"]
want_excel = args["excel"]

# ────── fuzzy column mapping with safe unpack ─────────────────────
tokens = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql))
for tok in tokens:
    if tok.lower() in COLUMNS:
        continue
    best = process.extractOne(tok, COLUMNS,
                              scorer=fuzz.WRatio,
                              score_cutoff=65)          # ↓ a bit looser
    if best is None:
        st.warning(
            f"🤔 I’m not sure what **{tok}** refers to. "
            "Please rephrase or use one of these columns: "
            f"{', '.join(sorted(COLUMNS)[:20])} …"
        )
        st.stop()
    match, score, _ = best
    sql = re.sub(rf"\b{tok}\b", match, sql, flags=re.I)

st.code(sql, language="sql")

# ───────── run SQL ────────────────────────────────────────────────
try:
    result = duckdb.query_df(roof, "roof", sql).df()
except Exception as e:
    st.error(f"⛔ SQL error: {e}")
    st.stop()

if result.empty:
    st.warning("No rows matched that query.")
    st.stop()

# ───────── display ────────────────────────────────────────────────
def hilite(row):
    return [
        "background-color:#d2ead2"
        if c.endswith("_num") and pd.notna(row[c]) and row[c] <= 1 else ""
        for c in row.index
    ]

st.dataframe(result.style.apply(hilite, axis=1), use_container_width=True)

# ───────── optional Excel download ───────────────────────────────
if want_excel:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        result.to_excel(xl, sheet_name="RoofWindows", index=False)
    st.download_button(
        "⬇️ Download filtered Excel",
        data=buf.getvalue(),
        file_name="roof_windows_uk.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
# ------------------------------------------------------------------
