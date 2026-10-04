import streamlit as st
import pandas as pd
import google.generativeai as genai

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="VendorWise AI — Procurement Recommender",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ VendorWise AI — Strategic Procurement & Vendor Recommender")
st.caption("Hybrid Decision-Support System: Deterministic MCDA Scoring + Gemini 1.5 Flash Qualitative Audit")

# ------------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION & API KEY
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    api_key_input = st.text_input("Enter Gemini API Key:", type="password", help="Get free key at aistudio.google.com")
    api_key = st.secrets.get("GEMINI_API_KEY", api_key_input)
    
    st.divider()
    st.header("🎯 Hard Business Constraints")
    max_budget = st.number_input("Maximum Allowed Budget ($):", min_value=1000, value=50000, step=1000)
    min_sla = st.slider("Minimum Required SLA (%):", min_value=90.0, max_value=99.9, value=95.0, step=0.1)

    st.divider()
    st.header("📊 Evaluation Criteria Weights")
    w_price = st.slider("Price Weight (%)", 0, 100, 40)
    w_sla = st.slider("SLA / Quality Weight (%)", 0, 100, 30)
    w_delivery = st.slider("Delivery Speed Weight (%)", 0, 100, 20)
    w_esg = st.slider("ESG / Compliance Weight (%)", 0, 100, 10)
    
    total_weight = w_price + w_sla + w_delivery + w_esg
    if total_weight != 100:
        st.error(f"⚠️ Total Weight must equal 100%! Current sum: **{total_weight}%**")

# ------------------------------------------------------------------------------
# 3. SAMPLE DATASET INITIALIZATION
# ------------------------------------------------------------------------------
default_vendors = [
    {
        "name": "Vendor A (Apex Solutions)",
        "price": 42000,
        "sla": 99.5,
        "delivery_days": 10,
        "esg_score": 88,
        "notes": "Includes 24/7 priority support and dedicated TAM. Contract includes standard 3% annual escalation clause. Payment terms: Net 30."
    },
    {
        "name": "Vendor B (Global Logistics Corp)",
        "price": 36000,
        "sla": 92.0,  # Below min_sla 95% -> Triggers guardrail disqualification
        "delivery_days": 25,
        "esg_score": 62,
        "notes": "Lowest upfront price. However, past performance shows frequent shipping delays. Penalty clause capped at only 1% of total contract value."
    },
    {
        "name": "Vendor C (TechPrime Enterprise)",
        "price": 54000, # Above budget $50,000 -> Triggers guardrail disqualification
        "sla": 99.9,
        "delivery_days": 7,
        "esg_score": 95,
        "notes": "Premium market leader. Zero downtime guarantee. Requires 50% upfront payment deposit and strict non-disclosure terms."
    }
]

if "vendors" not in st.session_state:
    st.session_state.vendors = default_vendors

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE — VENDOR INPUTS
# ------------------------------------------------------------------------------
st.subheader("📝 Vendor Proposals")
cols = st.columns(3)

for idx, col in enumerate(cols):
    v = st.session_state.vendors[idx]
    with col:
        with st.expander(f"📌 {v['name']}", expanded=True):
            v['price'] = st.number_input(f"Price ($) - V{idx+1}", value=v['price'], step=1000, key=f"p_{idx}")
            v['sla'] = st.number_input(f"SLA Uptime (%) - V{idx+1}", value=v['sla'], step=0.1, key=f"s_{idx}")
            v['delivery_days'] = st.number_input(f"Delivery Time (Days) - V{idx+1}", value=v['delivery_days'], step=1, key=f"d_{idx}")
            v['esg_score'] = st.slider(f"ESG Score (1-100) - V{idx+1}", 1, 100, v['esg_score'], key=f"e_{idx}")
            v['notes'] = st.text_area(f"Contract / Fine Print Notes - V{idx+1}", value=v['notes'], key=f"n_{idx}")

# ------------------------------------------------------------------------------
# 5. DETERMINISTIC RULE ENGINE & MCDA SCORING
# ------------------------------------------------------------------------------
st.divider()
st.subheader("⚡ Deterministic Decision Analysis")

if total_weight != 100:
    st.warning("Please adjust criteria weights in the sidebar to equal exactly 100% before evaluating.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(st.session_state.vendors)

# Step 1: Hard Constraint Evaluation
df['Budget_Pass'] = df['price'] <= max_budget
df['SLA_Pass'] = df['sla'] >= min_sla
df['Eligible'] = df['Budget_Pass'] & df['SLA_Pass']

# Step 2: Min-Max Normalization
def normalize(series, higher_is_better=True):
    if series.max() == series.min():
        return pd.Series(100.0, index=series.index)
    if higher_is_better:
        return (series - series.min()) / (series.max() - series.min()) * 100
    else:
        return (series.max() - series) / (series.max() - series.min()) * 100

norm_price = normalize(df['price'], higher_is_better=False)
norm_sla = normalize(df['sla'], higher_is_better=True)
norm_delivery = normalize(df['delivery_days'], higher_is_better=False)
norm_esg = normalize(df['esg_score'], higher_is_better=True)

# Calculate Composite Score (0 to 100)
df['Composite_Score'] = (
    (norm_price * (w_price / 100)) +
    (norm_sla * (w_sla / 100)) +
    (norm_delivery * (w_delivery / 100)) +
    (norm_esg * (w_esg / 100))
).round(2)

# Override Score for Ineligible Vendors
df.loc[~df['Eligible'], 'Composite_Score'] = 0.0

# Display Comparison Table
st.markdown("### 📊 Calculated Comparison & Compliance Leaderboard")
display_df = df[['name', 'price', 'sla', 'delivery_days', 'esg_score', 'Budget_Pass', 'SLA_Pass', 'Composite_Score']].copy()
display_df.columns = ['Vendor Name', 'Price ($)', 'SLA (%)', 'Delivery (Days)', 'ESG Score', 'Budget Check', 'SLA Check', 'MCDA Score (0-100)']

st.dataframe(display_df.style.highlight_max(axis=0, subset=['MCDA Score (0-100)'], color='lightgreen'), use_container_width=True)

# ------------------------------------------------------------------------------
# 6. AI GENERATIVE AUDIT & RECOMMENDATION ENGINE
# ------------------------------------------------------------------------------
st.divider()
st.subheader("🤖 AI Qualitative Risk Audit & Executive Playbook")

if not api_key:
    st.info("👈 Enter your free Gemini API Key in the sidebar to activate the AI analysis engine.")
else:
    if st.button("🚀 Generate Strategic Procurement Recommendation"):
        genai.configure(api_key=api_key)
        
        with st.spinner("Analyzing contract fine print and generating executive memorandum..."):
            try:
                vendors_summary = ""
                for idx, row in df.iterrows():
                    status = "ELIGIBLE" if row['Eligible'] else f"DISQUALIFIED (Budget Pass: {row['Budget_Pass']}, SLA Pass: {row['SLA_Pass']})"
                    vendors_summary += f"""
                    ---
                    Vendor Name: {row['name']}
                    Status: {status}
                    Calculated Composite Score: {row['Composite_Score']}/100
                    Price: ${row['price']:,} (Max Limit: ${max_budget:,})
                    SLA: {row['sla']}% (Min Required: {min_sla}%)
                    Delivery Time: {row['delivery_days']} days
                    ESG Score: {row['esg_score']}/100
                    Contract / Qualitative Notes: {row['notes']}
                    """

                system_prompt = f"""
                You are VendorWise AI, an expert Chief Procurement Officer (CPO) and Risk Auditor.
                Your task is to provide an executive vendor evaluation memo based strictly on the structured data provided below.

                SYSTEM GUARDRAILS:
                1. You MUST NEVER recommend a vendor marked as DISQUALIFIED, regardless of how good their qualitative notes are.
                2. Explicitly validate that the math and rule-based compliance checks were completed by the system.
                3. Structure your response into 3 clear sections:
                   - SECTION 1: Executive Winner & Rationale
                   - SECTION 2: Contract Red-Flag & Hidden Risk Analysis
                   - SECTION 3: Tactical Negotiation Playbook

                --- VENDOR EVALUATION DATA ---
                {vendors_summary}
                -------------------------------
                """
model = genai.GenerativeModel("gemini-2.0-flash")
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(system_prompt)
                
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error executing AI Analysis: {e}")
