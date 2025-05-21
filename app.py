import os, io, re, duckdb, streamlit as st, pandas as pd
from dotenv import load_dotenv
load_dotenv()                  # allows local testing with a .env file

st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───── Sidebar branding ──────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)
st.sidebar.markdown("### Roof-Window Knowledge-Bot\nUK market – Proof of Concept")

for q in (
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide"
):
    if st.sidebar.button(q):
        st.session_state["prompt"] = q

# ───── Load cleaned data only once ───────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/roof_windows_uk.parquet")
roof = load_data()

# ───── User prompt ───────────────────────────────────────────────────
prompt = st.text_input("Ask a question about UK roof-windows:", key="prompt")

# ───── When user submits question ────────────────────────────────────
if prompt:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    SYSTEM = """
You are a data assistant. Answer only by calling the function. Keys:
  sql   : SQL that answers the question using table roof
  excel : true if user wants a downloadable Excel
"""
    context = f"User: {prompt}\nColumns: {', '.join(roof.columns[:20])} …"

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",   "content": context}],
        tools=[{
            "type": "function",
            "function": {
                "name": "execute_query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "excel": {"type": "boolean"}
                    },
                    "required": ["sql", "excel"]
                }
            }}],
        tool_choice="auto"
    )

    tool_args = response.choices[0].message.tool_calls[0].function.arguments
    sql, wants_excel = tool_args["sql"], tool_args["excel"]

    with st.spinner("Crunching…"):
        try:
            result = duckdb.query_df(roof, "roof", sql).df()
        except Exception as e:
            st.error(f"⛔ SQL error: {e}")
            st.code(sql, language="sql")
            st.stop()

    # Colour cells where *_num ≤ 1.0
    def hilite(row):
        style = []
        for c in row.index:
            if c.endswith("_num") and pd.notna(row[c]) and row[c] <= 1:
                style.append("background-color:#d2ead2")
            else:
                style.append("")
        return style

    st.dataframe(result.style.apply(hilite, axis=1),
                 use_container_width=True)

    if wants_excel:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
            result.to_excel(xl, sheet_name="RoofWindows", index=False)
            ws = xl.sheets["RoofWindows"]
            green = xl.book.add_format({"bg_color": "#d2ead2"})
            for col_num, col in enumerate(result.columns):
                if col.endswith("_num"):
                    ws.conditional_format(1, col_num, len(result), col_num,
                                          {"type": "cell", "criteria": "<=", "value": 1, "format": green})
        st.download_button("⬇️ Download filtered Excel",
                           data=buf.getvalue(),
                           file_name="roof_windows_uk.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
