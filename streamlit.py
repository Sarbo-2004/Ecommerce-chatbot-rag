import streamlit
from rag_pipeline import ask_question
 
streamlit.title("Tesla Financial RAG Chatbot (Gemini)")
 
query = streamlit.text_input("Ask a question about Tesla Financial Data")
 
if streamlit.button("Ask"):
    if query:
        with streamlit.spinner("Thinking..."):
            result = ask_question(query)
       
        streamlit.subheader("Answer")
        streamlit.write(result["answer"])
 
        streamlit.subheader("Sources")
        for src in result["sources"]:
            streamlit.write("-", str(src)[:200], "...")
 
 