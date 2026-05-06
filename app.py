import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama
import json
import re
import traceback

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Analyst",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .chat-bubble-user {
        background: #e8f0fe;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 6px 0;
        margin-left: 40px;
    }
    .chat-bubble-ai {
        background: #f1f3f4;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px;
        margin: 6px 0;
        margin-right: 40px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Business Analyst")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_chart_plan" not in st.session_state:
    st.session_state.last_chart_plan = None
if "active_file" not in st.session_state:
    st.session_state.active_file = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: COLUMN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_columns(df):
    """Returns dicts of column types for smart chart planning."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = []

    # Try to detect datetime cols (already parsed or parseable)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif col in cat_cols:
            # try parsing as date
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    date_cols.append(col)
                    cat_cols.remove(col)
            except Exception:
                pass

    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "date": date_cols,
        "all": df.columns.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: SAFE DATE PARSE
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_datetime(df, col):
    """Return df copy with col coerced to datetime if not already."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: FIND CLOSEST COLUMN
# ═══════════════════════════════════════════════════════════════════════════════

def find_closest_column(name, columns):
    """Case-insensitive fuzzy match for column names."""
    if not name:
        return None
    name_lower = name.lower().strip()
    for col in columns:
        if col.lower().strip() == name_lower:
            return col
    for col in columns:
        if name_lower in col.lower() or col.lower() in name_lower:
            return col
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: EXTRACT JSON FROM LLM RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json(text):
    """Try multiple strategies to parse JSON from LLM text."""
    # Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Extract first JSON block
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Look for ```json blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: LLAMA3 CHART PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def plan_chart(request, df, col_info):
    """Call Llama3 to return a structured chart plan as JSON."""

    numeric_str = ", ".join(col_info["numeric"]) or "none"
    cat_str = ", ".join(col_info["categorical"]) or "none"
    date_str = ", ".join(col_info["date"]) or "none"

    system_prompt = f"""You are a BI chart planning engine. Given a user request and dataset metadata, return ONLY a JSON object with chart instructions. No explanation, no markdown, no extra text — ONLY the JSON.

Dataset columns:
- Numeric columns: {numeric_str}
- Categorical columns: {cat_str}
- Date/Time columns: {date_str}
- All columns: {", ".join(col_info["all"])}

Supported chart_type values: bar, line, pie, scatter, histogram, box, area, heatmap, stacked_bar, grouped_bar

JSON format to return:
{{
  "chart_type": "bar",
  "x": "column_name_or_empty",
  "y": "column_name_or_empty",
  "color": "column_name_or_empty",
  "aggregation": "sum|count|mean|none",
  "group_by": "column_name_or_empty",
  "time_granularity": "month|year|quarter|day|none",
  "top_n": 0,
  "sort_desc": true,
  "filter_col": "",
  "filter_val": "",
  "title": "Chart Title Here"
}}

Rules:
- For "top N" requests set top_n to N (e.g. 10 for top 10)
- For time trends use the date column as x, set time_granularity
- For heatmap correlation use chart_type heatmap, x and y can be empty (will use all numeric)
- For scatter use two numeric columns for x and y
- For histogram use one numeric column as x, y should be empty
- For box plot x=categorical, y=numeric
- For stacked_bar/grouped_bar set color to the second grouping column
- aggregation applies to y column: sum, count, or mean
- If user says "count" or categorical only, use aggregation=count
- Always set a good descriptive title
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {request}"},
        ],
    )

    raw = response["message"]["content"]
    return extract_json(raw), raw


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: CHART EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def execute_chart(plan, df, col_info):
    """
    Execute a chart plan dict against a DataFrame.
    Returns (fig, debug_info) or raises Exception.
    """
    chart_type = (plan.get("chart_type") or "bar").lower().strip()
    x_raw = plan.get("x", "") or ""
    y_raw = plan.get("y", "") or ""
    color_raw = plan.get("color", "") or ""
    aggregation = (plan.get("aggregation") or "sum").lower()
    group_by_raw = plan.get("group_by", "") or ""
    time_granularity = (plan.get("time_granularity") or "none").lower()
    top_n = int(plan.get("top_n") or 0)
    sort_desc = plan.get("sort_desc", True)
    title = plan.get("title") or chart_type.replace("_", " ").title()
    filter_col = plan.get("filter_col", "") or ""
    filter_val = plan.get("filter_val", "") or ""

    columns = df.columns.tolist()

    # Resolve column names
    x = find_closest_column(x_raw, columns)
    y = find_closest_column(y_raw, columns)
    color = find_closest_column(color_raw, columns)
    group_by = find_closest_column(group_by_raw, columns)

    # ── Apply filter ──────────────────────────────
    if filter_col and filter_val:
        fc = find_closest_column(filter_col, columns)
        if fc:
            df = df[df[fc].astype(str).str.lower() == filter_val.lower()]

    # ── HEATMAP (correlation) ─────────────────────
    if chart_type == "heatmap":
        num_df = df[col_info["numeric"]].dropna(axis=1, how="all")
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for a heatmap.")
        corr = num_df.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title=title,
            aspect="auto",
        )
        return fig, "heatmap correlation"

    # ── HISTOGRAM ────────────────────────────────
    if chart_type == "histogram":
        col = x or (col_info["numeric"][0] if col_info["numeric"] else None)
        if not col:
            raise ValueError("No numeric column found for histogram.")
        fig = px.histogram(df, x=col, color=color if color else None, title=title)
        return fig, f"histogram of {col}"

    # ── SCATTER ──────────────────────────────────
    if chart_type == "scatter":
        if not x or not y:
            nums = col_info["numeric"]
            if len(nums) >= 2:
                x = x or nums[0]
                y = y or nums[1]
            else:
                raise ValueError("Need two numeric columns for scatter plot.")
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color if color else None,
            title=title,
            trendline="ols" if len(df) < 10000 else None,
        )
        return fig, f"scatter {x} vs {y}"

    # ── BOX PLOT ─────────────────────────────────
    if chart_type == "box":
        cat = x or (col_info["categorical"][0] if col_info["categorical"] else None)
        val = y or (col_info["numeric"][0] if col_info["numeric"] else None)
        if not val:
            raise ValueError("No numeric column for box plot.")
        fig = px.box(df, x=cat, y=val, color=color if color else None, title=title)
        return fig, f"box {val} by {cat}"

    # ── TIME-SERIES (LINE / AREA) ─────────────────
    if time_granularity and time_granularity != "none":
        date_col = x or (col_info["date"][0] if col_info["date"] else None)
        if date_col:
            dft = ensure_datetime(df, date_col)
            dft = dft.dropna(subset=[date_col])

            freq_map = {"month": "ME", "year": "YE", "quarter": "QE", "day": "D"}
            freq = freq_map.get(time_granularity, "ME")

            val_col = y or (col_info["numeric"][0] if col_info["numeric"] else None)

            if val_col:
                dft = dft.set_index(date_col)
                if aggregation == "count":
                    grouped = dft.resample(freq)[val_col].count().reset_index()
                    grouped.columns = [date_col, val_col]
                elif aggregation == "mean":
                    grouped = dft.resample(freq)[val_col].mean().reset_index()
                    grouped.columns = [date_col, val_col]
                else:
                    grouped = dft.resample(freq)[val_col].sum().reset_index()
                    grouped.columns = [date_col, val_col]

                if chart_type == "bar":
                    fig = px.bar(grouped, x=date_col, y=val_col, title=title)

                elif chart_type == "area":
                    fig = px.area(grouped, x=date_col, y=val_col, title=title)

                else:
                    fig = px.line(grouped, x=date_col, y=val_col, title=title, markers=True)
                return fig, f"{time_granularity} trend of {val_col}"

    # ── AGGREGATION CHARTS (bar / line / pie / stacked / grouped) ────────────
    grp = group_by or x or (col_info["categorical"][0] if col_info["categorical"] else None)
    val_col = y or (col_info["numeric"][0] if col_info["numeric"] else None)

    if not grp:
        raise ValueError("Could not determine grouping column.")

    # Determine secondary grouping (for stacked/grouped)
    color_grp = color or (group_by if group_by and group_by != grp else None)

    # Build aggregated dataframe
    if color_grp and color_grp in df.columns and color_grp != grp:
        group_cols = [grp, color_grp]
    else:
        group_cols = [grp]
        color_grp = None

    if aggregation == "count" or not val_col:
        agg_df = df.groupby(group_cols).size().reset_index(name="Count")
        y_col = "Count"
    elif aggregation == "mean":
        agg_df = df.groupby(group_cols)[val_col].mean().reset_index()
        agg_df.columns = group_cols + [val_col]
        y_col = val_col
    else:
        agg_df = df.groupby(group_cols)[val_col].sum().reset_index()
        agg_df.columns = group_cols + [val_col]
        y_col = val_col

    # Sort + top N (only on single-group charts or total)
    if not color_grp:
        agg_df = agg_df.sort_values(y_col, ascending=not sort_desc)
        if top_n and top_n > 0:
            agg_df = agg_df.head(top_n)
    else:
        # For multi-group: sort by total per primary group
        totals = agg_df.groupby(grp)[y_col].sum().sort_values(ascending=not sort_desc)
        if top_n and top_n > 0:
            totals = totals.head(top_n)
        agg_df = agg_df[agg_df[grp].isin(totals.index)]
        agg_df[grp] = pd.Categorical(agg_df[grp], categories=totals.index, ordered=True)
        agg_df = agg_df.sort_values(grp)

    # ── PIE ──────────────────────────────────────
    if chart_type == "pie":
        fig = px.pie(
            agg_df,
            names=grp,
            values=y_col,
            title=title,
            hole=0.3,
        )
        return fig, f"pie {y_col} by {grp}"

    # ── STACKED BAR ──────────────────────────────
    if chart_type == "stacked_bar" and color_grp:
        fig = px.bar(
            agg_df,
            x=grp,
            y=y_col,
            color=color_grp,
            barmode="stack",
            title=title,
        )
        return fig, f"stacked bar {y_col} by {grp}/{color_grp}"

    # ── GROUPED BAR ──────────────────────────────
    if chart_type == "grouped_bar" and color_grp:
        fig = px.bar(
            agg_df,
            x=grp,
            y=y_col,
            color=color_grp,
            barmode="group",
            title=title,
        )
        return fig, f"grouped bar {y_col} by {grp}/{color_grp}"

    # ── LINE ─────────────────────────────────────
    if chart_type == "line":
        fig = px.line(
            agg_df,
            x=grp,
            y=y_col,
            color=color_grp if color_grp else None,
            title=title,
            markers=True,
        )
        return fig, f"line {y_col} by {grp}"

    # ── AREA ─────────────────────────────────────
    if chart_type == "area":
        fig = px.area(
            agg_df,
            x=grp,
            y=y_col,
            color=color_grp if color_grp else None,
            title=title,
        )
        return fig, f"area {y_col} by {grp}"

    # ── DEFAULT: BAR ─────────────────────────────
    fig = px.bar(
        agg_df,
        x=grp,
        y=y_col,
        color=color_grp if color_grp else None,
        title=title,
        text_auto=True,
    )
    fig.update_traces(textposition="outside")
    return fig, f"bar {y_col} by {grp}"


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: CHAT ANSWER
# ═══════════════════════════════════════════════════════════════════════════════

def chat_with_data(question, df, col_info, history):
    """Send a question + context to Llama3, return answer string."""
    numeric_summary = ""
    if col_info["numeric"]:
        try:
            numeric_summary = df[col_info["numeric"]].describe().round(2).to_string()
        except Exception:
            pass

    history_str = ""
    for h in history[-6:]:
        role = "User" if h["role"] == "user" else "Analyst"
        history_str += f"{role}: {h['content']}\n"

    prompt = f"""You are a senior business analyst and data expert. Answer the user's question about the dataset clearly and insightfully.

Dataset overview:
- Rows: {len(df):,}
- Columns: {", ".join(col_info["all"])}
- Numeric columns: {", ".join(col_info["numeric"]) or "none"}
- Categorical columns: {", ".join(col_info["categorical"]) or "none"}
- Date columns: {", ".join(col_info["date"]) or "none"}
- Missing values: {int(df.isnull().sum().sum()):,}

Sample data (10 random rows):
{df.sample(min(10, len(df))).to_string(index=False)}

Numeric statistics:
{numeric_summary}

Conversation so far:
{history_str}

User's question: {question}

Instructions:
- Be specific and data-driven
- Mention actual column names and values when relevant
- Highlight trends, anomalies, or business risks if applicable
- Keep the response concise but insightful — like a real analyst presenting findings
- Use bullet points for lists of insights
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def generate_dynamic_examples(col_info):
    examples = []

    num = col_info["numeric"]
    cat = col_info["categorical"]
    date = col_info["date"]

    if cat and num:
        examples.append(f"top 10 {cat[0]} by {num[0]}")
        examples.append(f"{num[0]} by {cat[0]}")
        examples.append(f"pie chart of {cat[0]}")
        examples.append(f"box plot {num[0]} by {cat[0]}")

    if len(num) >= 2:
        examples.append(f"scatter {num[0]} vs {num[1]}")
        examples.append("heatmap correlations")

    if date and num:
        examples.append(f"monthly {num[0]} trend")
        examples.append(f"yearly {num[0]} trend")

    if num:
        examples.append(f"histogram of {num[0]}")

    return examples[:10]

# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

file = st.file_uploader("📂 Upload CSV / Excel File", type=["csv", "xlsx", "xls"])

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

if file:
    filename = file.name.lower()

    # ── Detect new file upload → reset all state ───────────────────────────
    file_id = f"{file.name}_{file.size}"
    if st.session_state.active_file != file_id:
        st.session_state.active_file = file_id
        st.session_state.chat_history = []
        st.session_state.last_chart_plan = None
        # Clear any widget state tied to previous dataset
        for key in list(st.session_state.keys()):
            if key.startswith("sug_") or key in ("chart_request", "chat_input", "col_filter"):
                del st.session_state[key]

    # ── Load File ──────────────────────────────────────────────────────────
    try:
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1")
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if df.empty:
        st.error("❌ Uploaded file is empty.")
        st.stop()

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Classify columns
    col_info = classify_columns(df)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 All Charts", "📄 Data"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:

        # KPI metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🗂️ Total Rows", f"{len(df):,}")
        c2.metric("📐 Total Columns", len(df.columns))
        c3.metric("⚠️ Missing Values", f"{int(df.isnull().sum().sum()):,}")
        miss_pct = round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 1)
        c4.metric("📉 Missing %", f"{miss_pct}%")

        st.divider()

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">👤 <strong>You</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bubble-ai">🤖 <strong>AI Analyst</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Input row
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            question = st.text_input(
                "Ask a question about your data",
                placeholder="e.g. summarize this data · what issues do you see · top insights · which region performs best",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_btn:
            ask_btn = st.button("Ask →", use_container_width=True)

        # Suggestion chips
        suggestions = [
            "Summarize this dataset",
            "What are the top insights?",
            "Are there any data quality issues?",
            "What trends do you see?",
            "Which columns are most important?",
        ]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
                question = sug
                ask_btn = True

        if ask_btn and question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("🧠 Analyzing your data..."):
                try:
                    answer = chat_with_data(question, df, col_info, st.session_state.chat_history)
                except Exception as e:
                    answer = f"⚠️ Could not connect to Ollama. Make sure llama3 is running locally.\n\nError: {e}"

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — ALL CHARTS (DYNAMIC ENGINE)
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:

        st.subheader("🤖 Describe any chart in natural language")

        examples = generate_dynamic_examples(col_info)

        st.caption(
            "**Examples:** " + " · ".join([f"`{e}`" for e in examples])
)

        chart_request = st.text_input(
            "Describe any chart you want",
            placeholder="e.g. top 10 customers by revenue, month over month sales trend, pie chart by region…",
            label_visibility="collapsed",
            key="chart_request",
        )

        col_gen, col_info_btn = st.columns([3, 1])
        with col_gen:
            generate_btn = st.button("📊 Generate Chart", use_container_width=True, type="primary")
        with col_info_btn:
            show_plan = st.checkbox("Show chart plan (debug)", value=False)

        if generate_btn and chart_request:

            with st.spinner("🧠 Planning chart with AI..."):
                try:
                    plan, raw_response = plan_chart(chart_request, df, col_info)
                except Exception as e:
                    st.error(f"⚠️ Could not connect to Ollama llama3. Make sure it is running.\n\nError: {e}")
                    st.stop()

            if plan is None:
                st.warning("⚠️ AI returned an unparseable response. Trying fallback…")
                # Fallback: simple bar chart on first categorical + numeric
                plan = {
                    "chart_type": "bar",
                    "x": col_info["categorical"][0] if col_info["categorical"] else col_info["all"][0],
                    "y": col_info["numeric"][0] if col_info["numeric"] else "",
                    "aggregation": "sum",
                    "top_n": 10,
                    "sort_desc": True,
                    "title": f"Top 10 {col_info['categorical'][0] if col_info['categorical'] else 'Items'} by {col_info['numeric'][0] if col_info['numeric'] else 'Count'}",
                }

            if show_plan:
                st.json(plan)

            # Try to execute, with one automatic retry
            for attempt in range(2):
                try:
                    fig, description = execute_chart(plan, df, col_info)

                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(size=13),
                        title_font_size=18,
                        margin=dict(t=60, l=40, r=40, b=40),
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"✅ Rendered: {description}")
                    st.session_state.last_chart_plan = plan
                    break

                except Exception as e:
                    if attempt == 0:
                        # Retry with simpler fallback plan
                        fallback_x = col_info["categorical"][0] if col_info["categorical"] else col_info["all"][0]
                        fallback_y = col_info["numeric"][0] if col_info["numeric"] else None
                        plan = {
                            "chart_type": "bar",
                            "x": fallback_x,
                            "y": fallback_y or "",
                            "aggregation": "count" if not fallback_y else "sum",
                            "top_n": 10,
                            "sort_desc": True,
                            "title": f"Top 10 by {fallback_x}",
                        }
                    else:
                        st.error(
                            f"❌ Could not build chart after 2 attempts.\n\n"
                            f"**Suggestions:**\n"
                            f"- Column names in dataset: `{', '.join(col_info['all'][:8])}`\n"
                            f"- Try: `bar chart of {col_info['categorical'][0] if col_info['categorical'] else 'column'}`\n"
                            f"- Try: `top 10 by {col_info['categorical'][0] if col_info['categorical'] else 'category'}`\n\n"
                            f"Debug: {e}"
                        )

        # ── Column Reference ───────────────────────────────────────────────
        with st.expander("📋 Dataset column reference"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**📊 Numeric columns**")
                for col in col_info["numeric"]:
                    st.code(col, language=None)
            with c2:
                st.markdown("**🏷️ Categorical columns**")
                for col in col_info["categorical"]:
                    st.code(col, language=None)
            with c3:
                st.markdown("**📅 Date columns**")
                for col in col_info["date"] or ["(none detected)"]:
                    st.code(col, language=None)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — DATA
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:

        st.subheader("📄 Full Dataset")

        # Quick filters
        with st.expander("🔍 Filter columns to display"):
            selected_cols = st.multiselect(
                "Choose columns",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key="col_filter",
            )
        display_df = df[selected_cols] if selected_cols else df

        st.dataframe(display_df, use_container_width=True, height=500)

        row_count = len(display_df)
        st.caption(f"Showing {row_count:,} rows × {len(selected_cols)} columns")

        # Download
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="export.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #888;">
        <div style="font-size: 64px;">📂</div>
        <h3 style="color: #555; margin-top: 12px;">Upload a CSV or Excel file to get started</h3>
        <p>Supports .csv, .xlsx, .xls</p>
    </div>
    """, unsafe_allow_html=True)