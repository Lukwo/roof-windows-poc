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
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───── sidebar ────────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)
else:
    st.sidebar.markdown("### Roof-Window Knowledge-Bot\n_UK market – PoC_")

example_questions = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide",
    "Are there any models with an internal white finish?",
    "What are the installation pitch ranges for VELUX windows?",
]
for q in example_questions:
    if st.sidebar.button(q, key=f"example_q_{q}"):
        st.session_state["prompt"] = q
        st.session_state.chat = [{"role": "system", "content": SYSTEM_PROMPT}]

if st.sidebar.button("🔄 Reset chat"):
    keys_to_pop = ["prompt", "sql_query_from_ai", "query_result_df", "want_excel_download"]
    for key in keys_to_pop:
        st.session_state.pop(key, None)
    st.session_state.chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()

# ───── load data ───────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    data_file_path = "data/roof_windows_uk.parquet"
    try:
        df = pd.read_parquet(data_file_path)
        if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
            st.error(f"🚨 **Data Error:** The file '{data_file_path}' was loaded but appears to be empty or not a valid table.")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"🚨 **Error:** The data file '{data_file_path}' was not found. Please ensure it's in a 'data' subfolder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 **Error loading data:** An unexpected error occurred: {e}")
        return pd.DataFrame()

roof_df = load_data()

if roof_df.empty:
    st.warning("Data could not be loaded. The application cannot proceed.")
    st.stop()

COLUMNS = list(roof_df.columns)

# ───── AI System Prompt ───────────────────────────────────────────
COLUMNS_DESCRIPTIONS_GUIDE = """
- 'brand': Manufacturer (e.g., 'Velux', 'FAKRO'). User might say "who makes it?".
- 'name': Model name/code.
- 'external_width_mm_num': External width (mm). User: "how wide?", "width?". Convert cm to mm.
- 'external_height_mm_num': External height (mm). User: "how tall?", "height?". Convert cm to mm.
- 'internal_finish_colour': Inside frame color (e.g., 'White Painted', 'White Polyurethane'). User: "inside color?", "white painted?", "white finish?".
- 'gas': Gas fill (e.g., 'Argon', 'Krypton'). User: "what gas?".
- 'laminated': If internal pane is laminated. User: "safety glass?".
- 'light_transmittance_num': Visible light pass-through (ratio, e.g., 0.75 for 75%). User: "how much light?".
- 'u_value_window_num': U-value (W/m²K), lower is better insulation. User: "insulation level?".
- 'size_code': Manufacturer's size code, e.g., 'U8A', 'MK04'. User might ask for "U8A size".
"""

SYSTEM_PROMPT = f"""
You are a friendly and highly intelligent data assistant for UK roof windows.
Your goal is to help non-technical users find information from a pandas DataFrame called 'roof_df'.

Your instructions:
1. **Understand User Intent:** Analyze the user's question. For follow-up questions, consider the context of previous queries or results in the chat history.
2. **Map to Technical Columns:** Use column descriptions to map user terms to technical column names.
3. **Unit Conversion:** Convert units if needed (e.g., cm to mm, % to decimal for 'light_transmittance_num').
4. **Query Generation:** Generate SQL for 'roof_df'. Use `ILIKE '%value%'` for partial text matches, `=` for exact. For follow-ups, combine with previous query conditions if relevant (e.g., keep prior filters like size_code).
5. **Polite Refusal:** If unable to answer, say so.
6. **Output Format:** Return ONLY a function call (JSON) with 'sql' (string) and 'excel' (boolean).

Allowed columns: {', '.join(sorted(COLUMNS))}
Column descriptions: {COLUMNS_DESCRIPTIONS_GUIDE}
Table name: `roof_df`.
"""

if "chat" not in st.session_state or not st.session_state.chat or st.session_state.chat[0].get("role") != "system":
    if 'SYSTEM_PROMPT' not in globals():
        st.error("System prompt not defined.")
        st.stop()
    st.session_state.chat = [{"role": "system", "content": SYSTEM_PROMPT}]

# ───── User Input ─────────────────────────────────────────────────
user_prompt_value = st.session_state.get("prompt", "")
user_prompt = st.text_input("Ask a question about UK roof windows:", key="prompt_input", value=user_prompt_value)

if not user_prompt:
    st.info("Ask a question or select an example from the sidebar.")
    st.stop()

# Add user's message to chat history if it's new
if not st.session_state.chat or st.session_state.chat[-1].get("role") != "user" or st.session_state.chat[-1].get("content") != user_prompt:
    st.session_state.chat.append({"role": "user", "content": user_prompt})

# ───── Call OpenAI ─────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 **Error:** OPENAI_API_KEY not found. Please configure it in Streamlit secrets.")
    st.stop()
openai.api_key = openai_api_key

# Validate chat history to ensure all tool_calls have responses
def validate_chat_history(chat_history):
    validated_history = []
    pending_tool_call_ids = set()
    for msg in chat_history:
        validated_history.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                pending_tool_call_ids.add(tool_call.id)
        elif msg.get("role") == "tool":
            if msg.get("tool_call_id") in pending_tool_call_ids:
                pending_tool_call_ids.remove(msg["tool_call_id"])
    # Add dummy tool responses for unresponded tool calls
    for tool_call_id in pending_tool_call_ids:
        validated_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "execute_query",
            "content": json.dumps({"status": "error", "error_message": "Tool call not processed due to session state."})
        })
    return validated_history

# Fuzzy column mapping
def fuzzy_map_columns(sql_query, valid_columns):
    def replace_column(match):
        col = match.group(1)
        if col not in valid_columns:
            best_match, score, _ = process.extractOne(col, valid_columns, scorer=fuzz.WRatio)
            if score > 80:  # Adjust threshold as needed
                return f'roof_df.{best_match}'
        return f'roof_df.{col}'
    
    pattern = r'\broof_df\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
    return re.sub(pattern, replace_column, sql_query)

# Validate and prepare chat history
chat_history_for_api = validate_chat_history(st.session_state.chat)

try:
    with st.spinner("Processing your query..."):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history_for_api,
            tools=[{
                "type": "function",
                "function": {
                    "name": "execute_query",
                    "description": "Executes a SQL query and indicates if Excel download is needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "SQL query for 'roof_df'."},
                            "excel": {"type": "boolean", "description": "True if user wants Excel."},
                        },
                        "required": ["sql", "excel"],
                    },
                },
            }],
            tool_choice={"type": "function", "function": {"name": "execute_query"}},
        )
except RateLimitError:
    st.error("🛑 OpenAI API quota exhausted. Please try again later.")
    st.stop()
except Exception as e:
    st.error(f"🚨 OpenAI API call failed: {e}")
    st.stop()

assistant_message = response.choices[0].message
st.session_state.chat.append(assistant_message)

tool_calls = assistant_message.tool_calls
if not tool_calls:
    if assistant_message.content:
        st.warning(f"ℹ️ Assistant: {assistant_message.content}")
    else:
        st.error("AI assistant did not return an SQL query. Please rephrase.")
    st.stop()

# Process tool calls
for tool_call in tool_calls:
    if tool_call.function.name == "execute_query":
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            st.error("AI returned invalid JSON for the query.")
            tool_message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps({"status": "error", "error_message": "Invalid JSON arguments from AI."})
            }
            st.session_state.chat.append(tool_message)
            st.stop()

        sql_query_from_ai = args.get("sql")
        want_excel_download = args.get("excel", False)
        st.session_state["sql_query_from_ai"] = sql_query_from_ai
        st.session_state["want_excel_download"] = want_excel_download

        if not sql_query_from_ai:
            st.error("AI did not provide an SQL query.")
            tool_message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps({"status": "error", "error_message": "AI did not provide SQL query."})
            }
            st.session_state.chat.append(tool_message)
            st.stop()

        # Apply fuzzy column mapping
        final_sql_query = fuzzy_map_columns(sql_query_from_ai, COLUMNS)

        # Validate SQL columns
        allowed_columns = set(COLUMNS)
        used_columns = set(re.findall(r'\broof_df\.([a-zA-Z_][a-zA-Z0-9_]*)\b', final_sql_query))
        invalid_columns = used_columns - allowed_columns
        if invalid_columns:
            st.error(f"AI generated SQL with invalid columns: {invalid_columns}")
            tool_message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps({"status": "error", "error_message": f"Invalid columns: {invalid_columns}"})
            }
            st.session_state.chat.append(tool_message)
            st.stop()

        st.markdown("##### Generated SQL Query:")
        st.code(final_sql_query, language="sql")

        tool_response_content = ""
        try:
            query_result_df = duckdb.query_df(roof_df, "roof_df", final_sql_query).df()
            st.session_state["query_result_df"] = query_result_df

            if query_result_df.empty:
                st.warning("No data matched your query. Check if the requested values exist in the data.")
                tool_response_content = json.dumps({"status": "success", "message": "Query executed, no matching data found.", "rows_returned": 0})
            else:
                st.markdown("##### Query Results:")
                st.dataframe(query_result_df, use_container_width=True)
                tool_response_content = json.dumps({"status": "success", "message": "Query executed successfully.", "rows_returned": len(query_result_df)})

                if want_excel_download:
                    if len(query_result_df) > 10000:
                        st.warning("Result set too large for Excel download. Limiting to 10,000 rows.")
                        query_result_df = query_result_df.head(10000)
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        query_result_df.to_excel(writer, sheet_name="RoofWindowsData", index=False)
                    st.download_button(
                        label="⬇️ Download Results as Excel",
                        data=excel_buffer.getvalue(),
                        file_name="roof_windows_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"⛔ **SQL Execution Error:** {e}")
            st.markdown("**Problematic SQL Query:**")
            st.code(final_sql_query, language="sql")
            tool_response_content = json.dumps({"status": "error", "error_message": str(e)})

        # Add tool response to chat history
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": tool_response_content
        }
        st.session_state.chat.append(tool_message)

        if "error" in tool_response_content:
            st.stop()

    else:
        st.error(f"AI called an unexpected function: {tool_call.function.name}")
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": json.dumps({"status": "error", "error_message": "Unexpected function called by AI."})
        }
        st.session_state.chat.append(tool_message)
        st.stop()

# Clear prompt
if "prompt" in st.session_state:
    del st.session_state["prompt"]