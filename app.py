# app.py  –  Roof-Window Assistant
# ──────────────────────────────────────────────────────────────────
import os
import io
import re
import json
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
import openai
from rapidfuzz import process, fuzz

load_dotenv()

st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───── sidebar ────────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)
st.sidebar.markdown("### Roof-Window Knowledge-Bot\n_UK market – PoC_")

example_questions = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide",
    "Are there any models with an internal white finish?",
    "What are the installation pitch ranges for VELUX windows?",
]
for q in example_questions:
    if st.sidebar.button(q, key=f"example_q_{q}"): # Added unique keys for buttons
        st.session_state["prompt"] = q
        # If chat exists, append system prompt again if starting fresh with example
        if "chat" in st.session_state and st.session_state.chat and st.session_state.chat[0]["role"] != "system":
             # This scenario is less likely with current flow but good for robustness
             pass # System prompt should be first
        elif "chat" not in st.session_state:
            st.session_state.chat = [] # Initialize if reset somehow cleared it without re-init

if st.sidebar.button("🔄 Reset chat"):
    keys_to_pop = ["chat", "prompt", "sql_query", "query_result_df", "want_excel"]
    for key in keys_to_pop:
        if key in st.session_state:
            st.session_state.pop(key, None)
    st.rerun()

# ───── load data ───────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    data_file_path = "data/roof_windows_uk.parquet"
    try:
        df = pd.read_parquet(data_file_path)
        # Basic validation: check if it's a DataFrame and has columns
        if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
            st.error(f"🚨 **Data Error:** The file '{data_file_path}' was loaded but appears to be empty or not a valid table. Please check the file content.")
            return pd.DataFrame() # Return empty DataFrame
        return df
    except FileNotFoundError:
        st.error(f"🚨 **Error:** The data file '{data_file_path}' was not found. Please make sure it's in a 'data' subfolder relative to your app.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 **Error loading data:** An unexpected error occurred while loading '{data_file_path}': {e}")
        return pd.DataFrame()

roof_df = load_data()

if roof_df.empty:
    st.warning("Data could not be loaded. The application cannot proceed without data.")
    st.stop()

COLUMNS = list(roof_df.columns)

# ───── AI System Prompt: Instructions for the AI ───────────────────
# !!! CRITICAL CUSTOMIZATION REQUIRED BELOW !!!
COLUMNS_DESCRIPTIONS_GUIDE = """
Here are descriptions of columns in the 'roof_df' table and common ways users might refer to them:

- 'brand': The manufacturer of the window (e.g., 'Velux', 'BETTER ENERGY', 'FAKRO'). Users might ask "who makes it?", "by what company?".
- 'name': The specific model name or product code of the window.
- 'external_width_mm_num': The external width of the window frame in millimeters (mm). Users might ask "how wide?", "width", "what is the breadth?". If they ask in cm, convert to mm (e.g., 78cm = 780mm).
- 'external_height_mm_num': The external height of the window frame in millimeters (mm). Users might ask "how tall?", "height?". If they ask in cm, convert to mm.
- 'internal_finish_colour': The color or finish of the window frame on the inside (e.g., 'White Polyurethane', 'Clear Lacquer Pine', 'PVC'). Users might ask "inside color?", "white frame?", "pine look?".
- 'gas': The type of inert gas used between the glass panes for insulation (e.g., 'Argon', 'Krypton'). Users might ask "what gas is inside?", "insulation gas?".
- 'laminated': Indicates if the internal pane is laminated for safety (e.g., 'Yes', 'No', True, False, or specific text like 'Laminated Inner Pane'). Users might ask "is it safety glass?", "laminated inside?".
- 'light_transmittance_num': A numerical value (often a percentage or ratio like 0.75) indicating how much visible light passes through the glass. Higher is more light. Users might ask "how much light comes through?", "brightness?", "lets in light?".
- 'u_value_window_num': The U-value (thermal transmittance) of the entire window in W/m²K. Lower U-value means better insulation. Users might ask "how good is the insulation?", "energy efficient?", "what's the U-value?".
- 'installation_roof_pitch_range_min_deg_num': The minimum recommended roof pitch in degrees for installing this window. Users might ask "minimum roof slope?", "suitable for low pitch roof?".
- 'installation_roof_pitch_range_max_deg_num': The maximum recommended roof pitch in degrees. Users might ask "maximum roof slope?", "can it go on a steep roof?".
- 'easy_to_clean_coating': Indicates if the external glass has an easy-to-clean coating. Users might ask "self-cleaning glass?".
- 'air_permeability_class_num': A class number indicating air tightness. Higher class can mean better air tightness.
- 'comments': General comments or notes about the product. Users might ask for 'notes' or 'additional details'.
- 'material': The primary material of the window frame (e.g., 'Pine Wood', 'PVC', 'Wood core with Polyurethane').

(*** IMPORTANT: Review and complete this list with YOUR actual column names and user-friendly descriptions. ***)
(*** Be very specific about units (mm, cm, degrees, W/m²K, percentages as decimals like 0.75 for 75%) ***)
(*** and instruct the AI on how to handle conversions if users ask in different units. ***)
"""

SYSTEM_PROMPT = f"""
You are a friendly and highly intelligent data assistant for information about UK roof windows.
Your goal is to help non-technical users find information from a pandas DataFrame called 'roof_df'.

Your instructions:
1.  **Understand User Intent:** Carefully analyze the user's question, even if it uses everyday language or non-technical terms.
2.  **Map to Technical Columns:** Use the column descriptions provided below to map the user's intent to the correct technical column names from the 'Allowed columns' list.
3.  **Unit Conversion:** If a user specifies a unit (e.g., "70 cm wide") and the relevant column uses a different unit (e.g., 'external_width_mm_num' which is in mm), you MUST perform the conversion (e.g., 70 cm = 700 mm) and use the converted value in your SQL query. If a percentage is mentioned like "75% light transmittance" and the column 'light_transmittance_num' stores it as a decimal (e.g., 0.75), convert "75%" to 0.75.
4.  **Query Generation:** Generate a SQL query for the 'roof_df' table.
    * Use `SELECT *` by default if the user doesn't specify columns, or select specific columns if appropriate to the question.
    * Apply filtering conditions in the `WHERE` clause based on the user's request.
    * For text comparisons where partial matches are useful (like brand names or descriptive text), use `ILIKE '%value%'` for case-insensitive substring matching. For exact matches on categorical text (like a specific color 'White Polyurethane'), use `column_name = 'Exact Value'`.
    * For numeric columns (usually ending in `_num`), use standard operators like `=`, `>`, `<`, `>=`, `<=`.
5.  **Polite Refusal:** If a user asks a question that cannot be answered with the available columns or is outside your capabilities, politely inform them. Do not attempt to guess or make up information.
6.  **Output Format:** Return ONLY a function call in JSON format with two properties:
    * `sql` (string): The SQL query.
    * `excel` (boolean): Set to `true` if the user explicitly asks to "download", "export", or receive an "Excel file". Otherwise, set to `false`.

Allowed columns in the 'roof_df' table:
{', '.join(sorted(COLUMNS))}

Column descriptions and common user terms:
{COLUMNS_DESCRIPTIONS_GUIDE}

Example of mapping and query generation:
User question: "Show me Velux windows that are white inside and wider than 75 cm."
Your interpretation for SQL might involve: brand ILIKE '%Velux%', internal_finish_colour = 'White Polyurethane', external_width_mm_num > 750. (Assuming 'White Polyurethane' is an exact category).
Resulting function call:
```json
{{
  "sql": "SELECT * FROM roof_df WHERE brand ILIKE '%Velux%' AND internal_finish_colour = 'White Polyurethane' AND external_width_mm_num > 750",
  "excel": false
}}
```

User question: "Which FAKRO models have a U-value less than 1.0? I want to download this."
Resulting function call:
```json
{{
  "sql": "SELECT * FROM roof_df WHERE brand ILIKE '%FAKRO%' AND u_value_window_num < 1.0",
  "excel": true
}}
```
If the user asks a vague question like "Tell me about windows", you can ask for clarification or provide a general overview based on a few key columns like brand, name, and a common feature.
Never invent new column names. Always use columns from the 'Allowed columns' list.
The table name to use in the SQL query is ALWAYS `roof_df`.
"""

if "chat" not in st.session_state or not st.session_state.chat or st.session_state.chat[0]["role"] != "system":
    st.session_state.chat = [{"role": "system", "content": SYSTEM_PROMPT}]

# ───── User Input ─────────────────────────────────────────────────
user_prompt = st.text_input("Ask a question about UK roof windows:", key="prompt", value=st.session_state.get("prompt", ""))

if not user_prompt:
    st.info("Ask a question using the input box above, or select an example from the sidebar.")
    st.stop()

# Add user's message to chat history, but only if it's new
if not st.session_state.chat or st.session_state.chat[-1].get("content") != user_prompt or st.session_state.chat[-1].get("role") != "user":
    st.session_state.chat.append({"role": "user", "content": user_prompt})


# ───── Call OpenAI ─────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 **Error:** OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop()
openai.api_key = openai_api_key

try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.chat,
        tools=[{
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Executes a SQL query against the roof window data and indicates if Excel download is needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql":   {"type": "string", "description": "The SQL query to execute on the 'roof_df' table."},
                        "excel": {"type": "boolean", "description": "True if user wants to download an Excel file."},
                    },
                    "required": ["sql", "excel"],
                },
            },
        }],
        tool_choice={"type": "function", "function": {"name": "execute_query"}},
    )
except RateLimitError:
    st.error("🛑 OpenAI API quota exhausted. Please check your OpenAI billing or API key, then try again.")
    st.stop()
except Exception as e:
    st.error(f"🚨 OpenAI API call failed: {e}")
    st.stop()

assistant_message = response.choices[0].message
if assistant_message not in st.session_state.chat: # Avoid duplicating if already added
    st.session_state.chat.append(assistant_message)

if not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0:
    if assistant_message.content:
        st.warning(f"ℹ️ Assistant: {assistant_message.content}")
    else:
        st.error("The AI assistant did not return the expected SQL query structure. Please try rephrasing.")
    st.stop()

tool_call = assistant_message.tool_calls[0]
if tool_call.function.name != "execute_query":
    st.error(f"The AI assistant called an unexpected function: {tool_call.function.name}")
    st.stop()

try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    st.error("The AI assistant returned invalid JSON arguments for the query. Please try again.")
    st.stop()

sql_query_from_ai = args.get("sql")
want_excel_download = args.get("excel", False)
st.session_state["sql_query"] = sql_query_from_ai # Store for potential reuse/display
st.session_state["want_excel"] = want_excel_download

if not sql_query_from_ai:
    st.error("The AI assistant did not provide an SQL query. Please check your question or try rephrasing.")
    st.stop()

# ───── Safe Fuzzy Column Mapping (Post-AI Fallback) ───────────────
corrected_sql_query = sql_query_from_ai
tokens_in_sql = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", sql_query_from_ai)) # Min 3 chars for a token

TABLE_NAME_IN_SQL = 'roof_df'
SQL_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'GROUP', 'BY', 'ORDER', 'LIMIT', 'AS', 'ON',
    'LEFT', 'JOIN', 'INNER', 'RIGHT', 'DESC', 'ASC', 'IS', 'NOT', 'NULL', 'LIKE', 'ILIKE', 'IN',
    'BETWEEN', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'UNION', 'ALL',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'HAVING', 'CAST', 'VARCHAR', 'INTEGER', 'NUMERIC', 'TEXT', 'REAL'
}
column_lookup_lowercase = {col.lower(): col for col in COLUMNS}

for tok in tokens_in_sql:
    tok_lower = tok.lower()
    tok_upper = tok.upper()

    if tok_lower == TABLE_NAME_IN_SQL.lower() or tok_upper in SQL_KEYWORDS:
        continue
    if "'" in tok or '"' in tok: # Skip tokens that are likely string literals
        continue

    if tok_lower in column_lookup_lowercase:
        canonical_column_name = column_lookup_lowercase[tok_lower]
        if tok != canonical_column_name:
            corrected_sql_query = re.sub(rf"\b{re.escape(tok)}\b", canonical_column_name, corrected_sql_query)
        continue

    best_match_info = process.extractOne(tok, COLUMNS, scorer=fuzz.WRatio, score_cutoff=80) # Stricter cutoff

    if best_match_info:
        matched_column_name = best_match_info[0]
        # st.info(f"ℹ️ Fuzzy mapping: Assuming '{tok}' refers to '{matched_column_name}'.")
        corrected_sql_query = re.sub(rf"\b{re.escape(tok)}\b", matched_column_name, corrected_sql_query, flags=re.IGNORECASE)
    # else:
        # If no good fuzzy match, assume it's a value or something else the AI intended.
        # The SQL execution will fail if it's an invalid column name.
        # print(f"DEBUG (Fuzzy Map): Token '{tok}' not mapped.") # For debugging

final_sql_query = corrected_sql_query
st.session_state["sql_query"] = final_sql_query # Update with corrected query
# ───────────────────────────────────────────────────────────────────

st.markdown("##### Generated SQL Query:")
st.code(final_sql_query, language="sql")

# ───── Execute SQL Query ──────────────────────────────────────────
try:
    query_result_df = duckdb.query_df(roof_df, TABLE_NAME_IN_SQL, final_sql_query).df()
    st.session_state["query_result_df"] = query_result_df
except Exception as e:
    st.error(f"⛔ **SQL Execution Error:** {e}")
    st.error("This might be due to an issue in the generated SQL query or an unrecognized term. Please try rephrasing your question or check the column descriptions in the system prompt if you are the developer.")
    st.stop()

if query_result_df.empty:
    st.warning("No data matched your query.")
    st.stop()

# ───── Display Results ────────────────────────────────────────────
st.markdown("##### Query Results:")

def highlight_low_u_value(row): # Example highlighter
    styles = [''] * len(row)
    if 'u_value_window_num' in row.index and pd.notna(row['u_value_window_num']):
        try:
            if float(row['u_value_window_num']) < 1.0:
                u_value_col_idx = row.index.get_loc('u_value_window_num')
                styles[u_value_col_idx] = 'background-color: #d2ead2; font-weight: bold;'
        except ValueError:
            pass # Not a float, ignore
    return styles

st.dataframe(query_result_df.style.apply(highlight_low_u_value, axis=1), use_container_width=True)

# ───── Optional Excel Download ────────────────────────────────────
if want_excel_download:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        query_result_df.to_excel(writer, sheet_name="RoofWindowsData", index=False)
    
    st.download_button(
        label="⬇️ Download Results as Excel",
        data=excel_buffer.getvalue(),
        file_name="roof_windows_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# For debugging or transparency, show chat history
# with st.expander("View Full Chat History with AI"):
#     for i, message in enumerate(st.session_state.chat):
#         with st.chat_message(message["role"]):
#             if message["role"] == "system" and i==0: # Only show system prompt once
#                 with st.popover("System Prompt (click to expand)"):
#                     st.markdown(f"<small>{message['content'][:500]}...</small>", unsafe_allow_html=True)
#             elif message["role"] != "system": # Don't show system prompt repeatedly
#                 st.write(message.get("content"))
#                 if message.get("tool_calls"):
#                     st.json(message['tool_calls'][0].function.arguments)
