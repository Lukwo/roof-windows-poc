# app.py  ·  Roof-Window Assistant  ·  UK PoC
# ---------------------------------------------------------------
# copy–paste the whole file (no extra blank lines above or below)

import os, io, re, json
import duckdb, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError

load_dotenv()                                  # allows local .env testing
st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ─────────── 0 · sidebar branding & sample prompts ────────────
LOGO = "logo.png"
if os.path.exists(LOGO):
    st.sidebar.image(LOGO, width=160)

st.sidebar.markdown(
    "### Roof-Window Knowledge-Bot\n"
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

# ─────────── 1 · load the cleaned parquet (cached) ────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/roof_windows_uk.parquet")

roof = load_data()

# ─────────── 2 · prompt box ───────────────────────────────────
prompt = st.text_input("Ask a question about UK roof windows:", key="prompt")

# stop early if nothing asked
if not prompt:
    st.stop()

# ─────────── 3 · call OpenAI with tool-calling ────────────────
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM = """
You are a strict data assistant.  Answer ONLY with the function call below.
Return JSON with keys:
  sql   – a SELECT statement that answers the question using table 'roof'
  excel – true if user wants a downloadable Excel file, else false
Never add extra keys or free-form text.
"""

try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",                      # change to gpt-3.5-turbo-0125 if desired
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"{prompt}\n\nColumns: {', '.join(roof.columns)}"}
        ],
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
    st.error("🛑 OpenAI quota exhausted for this API key. "
             "Top-up billing *or* paste a different key under **Settings → Secrets**, "
             "then click **Rerun**.")
    st.stop()

# -------- parse the tool arguments (JSON string ➜ dict) -------
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

sql         = args["sql"]
wants_excel = args["excel"]

# ─────────── 4 · execute SQL safely with DuckDB ───────────────
with st.spinner("Crunching data…"):
    try:
        result = duckdb.query_df(roof, "roof", sql).df()
    except Exception as e:
        st.error(f"⛔ SQL error: {e}")
        st.code(sql, language="sql")
        st.stop()

# ─────────── 5 · display the result in the app ────────────────
def highlight(row):
    """Green background for *_num ≤ 1.0"""
    out = []
    for c in row.index:
        if c.endswith("_num") and pd.notna(row[c]) and row[c] <= 1:
            out.append("background-color:#d2ead2")
        else:
            out.append("")
    return out

st.dataframe(
    result.style.apply(highlight, axis=1),
    use_container_width=True
)

# ─────────── 6 · optional Excel download ──────────────────────
if wants_excel:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as xl:
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
        data=buffer.getvalue(),
        file_name="roof_windows_uk.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------------------------------------------------------------
# end of file
