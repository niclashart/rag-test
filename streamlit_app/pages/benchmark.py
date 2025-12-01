"""Benchmark page for RAGAS evaluation."""
import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def get_headers():
    """Get headers (no authentication needed)."""
    return {}

def show_benchmark():
    """Show benchmark page."""
    st.title("📊 RAGAS Benchmarking")
    st.markdown("Evaluieren Sie Ihre RAG-Pipeline mit RAGAS-Metriken")
    
    # Initialize session state
    if "benchmark_questions" not in st.session_state:
        st.session_state.benchmark_questions = [""]
        st.session_state.benchmark_answers = [""]
        st.session_state.benchmark_contexts = [[""]]
        st.session_state.benchmark_ground_truths = [""]
    
    # Question input
    st.subheader("Benchmark-Daten eingeben")
    
    num_questions = st.number_input(
        "Anzahl der Fragen",
        min_value=1,
        max_value=10,
        value=len(st.session_state.benchmark_questions),
        step=1
    )
    
    # Adjust lists to match num_questions
    while len(st.session_state.benchmark_questions) < num_questions:
        st.session_state.benchmark_questions.append("")
        st.session_state.benchmark_answers.append("")
        st.session_state.benchmark_contexts.append([""])
        st.session_state.benchmark_ground_truths.append("")
    
    while len(st.session_state.benchmark_questions) > num_questions:
        st.session_state.benchmark_questions.pop()
        st.session_state.benchmark_answers.pop()
        st.session_state.benchmark_contexts.pop()
        st.session_state.benchmark_ground_truths.pop()
    
    # Input forms
    for i in range(num_questions):
        with st.expander(f"Frage {i+1}", expanded=i==0):
            st.session_state.benchmark_questions[i] = st.text_input(
                "Frage",
                value=st.session_state.benchmark_questions[i],
                key=f"question_{i}"
            )
            
            st.session_state.benchmark_answers[i] = st.text_area(
                "Antwort",
                value=st.session_state.benchmark_answers[i],
                key=f"answer_{i}",
                height=100
            )
            
            # Contexts
            st.write("Kontexte:")
            num_contexts = st.number_input(
                "Anzahl Kontexte",
                min_value=1,
                max_value=5,
                value=len(st.session_state.benchmark_contexts[i]),
                key=f"num_contexts_{i}"
            )
            
            # Adjust contexts list
            while len(st.session_state.benchmark_contexts[i]) < num_contexts:
                st.session_state.benchmark_contexts[i].append("")
            while len(st.session_state.benchmark_contexts[i]) > num_contexts:
                st.session_state.benchmark_contexts[i].pop()
            
            for j in range(num_contexts):
                st.session_state.benchmark_contexts[i][j] = st.text_area(
                    f"Kontext {j+1}",
                    value=st.session_state.benchmark_contexts[i][j],
                    key=f"context_{i}_{j}",
                    height=80
                )
            
            st.session_state.benchmark_ground_truths[i] = st.text_area(
                "Ground Truth (optional)",
                value=st.session_state.benchmark_ground_truths[i],
                key=f"ground_truth_{i}",
                height=80,
                help="Die erwartete Antwort (optional)"
            )
    
    # Run benchmark button
    if st.button("Benchmark ausführen", use_container_width=True, type="primary"):
        # Filter empty entries
        questions = [q for q in st.session_state.benchmark_questions if q.strip()]
        answers = [a for a in st.session_state.benchmark_answers if a.strip()]
        contexts = [
            [c for c in ctx if c.strip()]
            for ctx in st.session_state.benchmark_contexts
            if any(c.strip() for c in ctx)
        ]
        ground_truths = [
            gt for gt in st.session_state.benchmark_ground_truths
            if gt.strip()
        ] or None
        
        if len(questions) != len(answers) or len(questions) != len(contexts):
            st.error("Bitte füllen Sie alle Felder aus (Frage, Antwort, Kontexte)")
        else:
            with st.spinner("Benchmark wird ausgeführt..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/benchmark/run",
                        json={
                            "questions": questions,
                            "answers": answers,
                            "contexts": contexts,
                            "ground_truths": ground_truths
                        },
                        headers=get_headers()
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.benchmark_results = results
                        st.success("Benchmark erfolgreich ausgeführt!")
                        st.rerun()
                    else:
                        st.error(f"Fehler: {response.json().get('detail', 'Unbekannter Fehler')}")
                except Exception as e:
                    st.error(f"Fehler: {str(e)}")
    
    # Display results
    if "benchmark_results" in st.session_state:
        st.divider()
        st.subheader("Ergebnisse")
        
        results = st.session_state.benchmark_results
        summary = results.get("summary", {})
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Faithfulness",
                f"{summary.get('faithfulness', 0)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Answer Relevancy",
                f"{summary.get('answer_relevancy', 0)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Context Precision",
                f"{summary.get('context_precision', 0)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Context Recall",
                f"{summary.get('context_recall', 0)*100:.1f}%"
            )
        
        # Detailed results
        with st.expander("Detaillierte Ergebnisse"):
            import pandas as pd
            df = pd.DataFrame(results.get("results", []))
            st.dataframe(df)
        
        # Plot link
        if results.get("plot_path"):
            st.info(f"Vollständiges Dashboard: {results['plot_path']}")

