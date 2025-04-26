import streamlit as st
from main import build_graph
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Dual-Agent AI System", layout="centered")

st.title("🔍 Dual-Agent Research + Drafting Assistant")
st.markdown("This app uses **LangGraph**, **OpenAI**, and **Tavily** to search and write high-quality answers.")

question = st.text_input("Enter your research question", placeholder="E.g., How is AI used in agriculture?")

if st.button("Run Research + Draft"):
    if question:
        with st.spinner("Running dual agents..."):
            try:
                graph = build_graph()
                result = graph.invoke({"question": question})
                st.success("✅ Done!")

                st.subheader("🔎 Research Result")
                st.markdown(result['research_result'])

                st.subheader("📝 Drafted Answer")
                st.markdown(result['drafted_answer'])

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
