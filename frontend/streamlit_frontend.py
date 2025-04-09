import streamlit as st
import requests
import json

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("SHL Assessment Recommendation Tool")
st.markdown("Enter a job description or query below to get recommended SHL assessments:")


query = st.text_area("Enter Job Description or Query:", height=150)

# Configuration for backend API
BACKEND_URL = "http://localhost:8001"  # Update this to your deployed URL when ready
# BACKEND_URL = "https://your-render-url.onrender.com"  

# Example queries
st.sidebar.header("Example Queries")
example_queries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
    "I need assessments for an analyst role combining cognitive and personality tests, maximum 45 minutes."
]

for i, eq in enumerate(example_queries):
    if st.sidebar.button(f"Example {i+1}", key=f"ex_{i}"):
        query = eq
        st.session_state.query = eq

# Submit button
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json={"query": query}
                )
                
                if res.status_code == 200:
                    results = res.json().get("recommended_assessments", [])
                    if results:
                        st.success(f"Found {len(results)} recommended assessments")
                        
                        # Display results in a modern format
                        for i, r in enumerate(results, 1):
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown(f"### {i}.")
                                    st.markdown(f"**Duration:** {r['duration']} min")
                                
                                with col2:
                                    st.markdown(f"### [{r.get('name', 'Assessment')}]({r['url']})")
                                    st.markdown(f"{r['description']}")
                                    
                                   
                                    meta1, meta2 = st.columns(2)
                                    with meta1:
                                        st.markdown(f"**Remote:** {r['remote_support']}")
                                    with meta2:
                                        st.markdown(f"**Adaptive:** {r['adaptive_support']}")
                                    
                                    
                                    st.markdown(f"**Types:** {', '.join(r['test_type'])}")
                                
                                st.markdown("---")
                    else:
                        st.info("No recommendations found for your query.")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# Show health status
try:
    health = requests.get(f"{BACKEND_URL}/health")
    if health.status_code == 200:
        st.sidebar.success("API Status: Healthy")
    else:
        st.sidebar.error("API Status: Unhealthy")
except:
    st.sidebar.error("API Status: Unreachable")