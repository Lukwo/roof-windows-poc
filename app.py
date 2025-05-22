# app.py  –  Roof-Window Assistant with safe fuzzy-column mapping
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
import openai # Ensure openai is imported
from rapidfuzz import process, fuzz

# Load environment variables (like your OPENAI_API_KEY)
load_dotenv()

# Page configuration
st.set_page_config(page_title="Roof-Window Assistant", page_icon="🪟")

# ───── sidebar ────────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=160)
st.sidebar.markdown("### Roof-Window Knowledge-Bot\n_UK market – PoC_")

# Example questions buttons
example_questions = [
    "Show all BETTER ENERGY roof windows",
    "Which windows use Krypton gas?",
    "Download an Excel of centre-pivot models ≥78 cm wide",
    "Are there any models with an internal white finish?",
    "What are the installation pitch ranges for VELUX windows?",
]
for q in example_questions:
    if st.sidebar.button(q):
        st.session_state["prompt"] = q

# Reset chat button
if st.sidebar.button("🔄 Reset chat"):
    # Clear relevant session state keys
    keys_to_pop = ["chat", "prompt"]
    for key in keys_to_pop:
        if key in st.session_state:
            st.session_state.pop(key, None)
    st.rerun()

# ───── load data ───────────────────────────────────────────────────
@st.cache_data # Cache the data loading for performance
def load_data() -> pd.DataFrame:
    try:
        return pd.read_parquet("data/roof_windows_uk.parquet")
    except FileNotFoundError:
        st.error("🚨 **Error:** The data file 'data/roof_windows_uk.parquet' was not found. Please make sure it's in the correct location.")
        st.stop()
        return pd.DataFrame() # Return empty DataFrame on error to prevent further issues

roof_df = load_data()
if roof_df.empty and "data/roof_windows_uk.parquet": # Check if empty due to earlier stop
    st.stop()

COLUMNS = list(roof_df.columns)

# ───── AI System Prompt: Instructions for the AI ───────────────────

# !!! CRITICAL CUSTOMIZATION REQUIRED BELOW !!!
# Provide clear, user-friendly descriptions for YOUR columns.
# This helps the AI understand non-technical user questions.
# Use the actual column names from your 'roof_windows_uk.parquet' file.
COLUMNS_DESCRIPTIONS_GUIDE = """
Here are descriptions of columns in the 'roof_df' table and common ways users might refer to them:

- 'brand': The manufacturer of the window (e.g., 'Velux', 'BETTER ENERGY', 'FAKRO'). Users might ask "who makes it?", "by what company?".
- 'name': The specific model name or product code of the window.
- 'external_width_mm_num': The external width of the window frame in millimeters (mm). Users might ask "how wide?", "width", "what is the breadth?". If they ask in cm, convert to mm (e.g., 78cm = 780mm).
- 'external_height_mm_num': The external height of the window frame in millimeters (mm). Users might ask "how tall?", "height?". If they ask in cm, convert to mm.
- 'internal_finish_colour': The color or finish of the window frame on the inside (e.g., 'White Polyurethane', 'Clear Lacquer Pine', 'PVC'). Users might ask "inside color?", "white frame?", "pine look?".
- 'gas': The type of inert gas used between the glass panes for insulation (e.g., 'Argon', 'Krypton'). Users might ask "what gas is inside?", "insulation gas?".
- 'laminated': Indicates if the internal pane is laminated for safety (e.g., 'Yes', 'No', True, False). Users might ask "is it safety glass?", "laminated inside?".
- 'light_transmittance_num': A numerical value (often a percentage or ratio) indicating how much visible light passes through the glass. Higher is more light. Users might ask "how much light comes through?", "brightness?", "lets in light?".
- 'u_value_window_num': The U-value (thermal transmittance) of the entire window in W/m²K. Lower U-value means better insulation. Users might ask "how good is the insulation?", "energy efficient?", "what's the U-value?".
- 'installation_roof_pitch_range_min_deg_num': The minimum recommended roof pitch in degrees for installing this window. Users might ask "minimum roof slope?", "suitable for low pitch roof?".
- 'installation_roof_pitch_range_max_deg_num': The maximum recommended roof pitch in degrees. Users might ask "maximum roof slope?", "can it go on a steep roof?".
- 'easy_to_clean_coating': Indicates if the external glass has an easy-to-clean coating. Users might ask "self-cleaning glass?".
- 'air_permeability_class_num': A class number indicating air tightness. Higher class can mean better air tightness.
- 'comments': General comments or notes about the product. Users might ask for 'notes' or 'additional details'.
- 'material': The primary material of the window frame (e.g., 'Pine Wood', 'PVC', 'Wood core with Polyurethane').

(*** ADD MORE DESCRIPTIONS FOR YOUR OTHER IMPORTANT COLUMNS HERE ***)
(*** Be as clear as possible. Explain units if applicable (e.g., mm, cm, degrees, percentages) ***)
(*** and how the AI should handle conversions if users ask in different units. ***)
"""

SYSTEM_PROMPT = f"""
You are a friendly and highly intelligent data assistant for information about UK roof windows.
Your goal is to help non-technical users find information from a pandas DataFrame called 'roof_df'.

Your instructions:
1.  **Understand User Intent:** Carefully analyze the user's question, even if it uses everyday language or non-technical terms.
2.  **Map to Technical Columns:** Use the column descriptions provided below to map the user's intent to the correct technical column names from the 'Allowed columns' list.
3.  **Unit Conversion:** If a user specifies a unit (e.g., "70 cm wide") and the relevant column uses a different unit (e.g., 'external_width_mm_num' which is in mm), you MUST perform the conversion (e.g., 70 cm = 700 mm) and use the converted value in your SQL query.
4.  **Query Generation:** Generate a SQL query for the 'roof_df' table.
    * Use `SELECT *` by default if the user doesn't specify columns, or select specific columns if appropriate to the question.
    * Apply filtering conditions in the `WHERE` clause based on the user's request.
    * For text comparisons, use `ILIKE` for case-insensitive matching (e.g., `brand ILIKE '%velux%'`). For exact matches on categories, use `=` (e.g. `internal_finish_colour = 'White Polyurethane'`).
    * For numeric columns (usually ending in `_num`), use standard operators like `=`, `>`, `<`, `>=`, `<=`.
5.  **Polite Refusal:** If a user asks a question that cannot be answered with the available columns or is outside your capabilities, politely inform them.
6.  **Output Format:** Return ONLY a function call in JSON format with two properties:
    * `sql` (string): The SQL query.
    * `excel` (boolean): Set to `true` if the user explicitly asks to "download", "export", or receive an "Excel file". Otherwise, set to `false`.

Allowed columns in the 'roof_df' table:
{', '.join(sorted(COLUMNS))}

Column descriptions and common user terms:
{COLUMNS_DESCRIPTIONS_GUIDE}

Example of mapping and query generation:
User question: "Show me Velux windows that are white inside and wider than 75 cm."
Your interpretation for SQL might involve: brand ILIKE '%Velux%', internal_finish_colour ILIKE '%White%', external_width_mm_num > 750.
Resulting function call:
```json
{{
  "sql": "SELECT * FROM roof_df WHERE brand ILIKE '%Velux%' AND internal_finish_colour ILIKE '%White%' AND external_width_mm_num > 750",
  "excel": false
}}