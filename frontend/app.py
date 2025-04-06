import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("📋 SHL Assessment Recommendation Tool")
st.markdown("Enter a job description or query below to get recommended SHL assessments:")

query = st.text_area("🔍 Enter Job Description or Query:")

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        try:
            res = requests.get("http://127.0.0.1:8000/recommend", params={"query": query})
            results = res.json()["results"]
            if results:
                for i, r in enumerate(results, 1):
                    st.markdown(f"### {i}. [{r['name']}]({r['url']})")
                    st.markdown(f"- **Duration**: {r['duration']}")
                    st.markdown(f"- **Remote**: {r['remote']} | **Adaptive**: {r['adaptive']}")
                    st.markdown(f"- **Type**: {r['type']}")
                    st.markdown("---")
            else:
                st.info("No recommendations found for the given query.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
