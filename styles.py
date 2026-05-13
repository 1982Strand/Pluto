import streamlit as st

def inject_styles():
    st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #eee; }
    .stTabs [data-baseweb="tab"] { font-weight: bold; font-size: 16px; }
    .big-value { font-size: 48px; font-weight: 700; margin: 0; }
    .return-pos { color: #2e7d32; font-size: 20px; font-weight: 600; }
    .return-neg { color: #d32f2f; font-size: 20px; font-weight: 600; }
    .market-badge {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        font-size: 14px; font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)