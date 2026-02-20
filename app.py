import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="MBTI Predictor", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTextArea textarea {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.5;
    }
    .personality-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .dimension-card {
        color: #000000;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('Topography Pattern Texture.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_models():
    models = {}
    vectorizer = None

    model_files = {
        'E/I': 'mbti_model_ie.joblib',
        'S/N': 'mbti_model_ns.joblib',
        'T/F': 'mbti_model_ft.joblib',
        'J/P': 'mbti_model_jp.joblib'
    }

    vectorizer_file = 'mbti_vectorizer.joblib'
    if os.path.exists(vectorizer_file):
        try:
            vectorizer = joblib.load(vectorizer_file)
        except Exception:
            pass

    for dimension, filename in model_files.items():
        if os.path.exists(filename):
            try:
                models[dimension] = joblib.load(filename)
            except Exception:
                pass

    return models, vectorizer
PERSONALITY_TYPES = {
    'INTJ': 'The Architect - Strategic, independent, and visionary',
    'INTP': 'The Thinker - Logical, innovative, and curious',
    'ENTJ': 'The Commander - Bold, strategic, and strong-willed',
    'ENTP': 'The Debater - Quick-witted, clever, and resourceful',
    'INFJ': 'The Advocate - Creative, insightful, and principled',
    'INFP': 'The Mediator - Empathetic, creative, and idealistic',
    'ENFJ': 'The Protagonist - Charismatic, inspiring, and natural leader',
    'ENFP': 'The Campaigner - Enthusiastic, creative, and sociable',
    'ISTJ': 'The Logistician - Practical, reliable, and methodical',
    'ISFJ': 'The Protector - Warm, responsible, and conscientious',
    'ESTJ': 'The Executive - Organized, practical, and decisive',
    'ESFJ': 'The Consul - Caring, social, and community-minded',
    'ISTP': 'The Virtuoso - Bold, practical, and experimental',
    'ISFP': 'The Adventurer - Flexible, charming, and artistic',
    'ESTP': 'The Entrepreneur - Energetic, perceptive, and spontaneous',
    'ESFP': 'The Entertainer - Spontaneous, energetic, and enthusiastic'
}


def predict_personality(text, models, vectorizer):
    if not models or not vectorizer:
        return None, "Models not loaded"

    try:
        text_features = vectorizer.transform([text])
        predictions = {}
        personality_type = ""

        for dimension, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(text_features)[0]
                    if len(proba) > 1:
                        predictions[dimension] = {
                            'probability': proba[1],
                            'prediction': 1 if proba[1] > 0.5 else 0
                        }
                    else:
                        predictions[dimension] = {
                            'probability': proba[0],
                            'prediction': 1 if proba[0] > 0.5 else 0
                        }
                else:
                    pred = model.predict(text_features)[0]
                    predictions[dimension] = {
                        'probability': 0.75 if pred == 1 else 0.25,
                        'prediction': int(pred)
                    }

                if dimension == 'E/I':
                    personality_type += 'I' if predictions[dimension]['prediction'] == 1 else 'E'
                elif dimension == 'S/N':
                    personality_type += 'S' if predictions[dimension]['prediction'] == 0 else 'N'
                elif dimension == 'T/F':
                    personality_type += 'T' if predictions[dimension]['prediction'] == 0 else 'F'
                elif dimension == 'J/P':
                    personality_type += 'J' if predictions[dimension]['prediction'] == 1 else 'P'

            except Exception:
                predictions[dimension] = {'probability': 0.5, 'prediction': 0}
                personality_type += 'X'

        return {
            'personality_type': personality_type,
            'dimensions': predictions,
            'description': PERSONALITY_TYPES.get(personality_type, 'Unknown personality type')
        }, None

    except Exception as e:
        return None, f"Prediction error: {str(e)}"
def create_personality_radar(predictions):
    """Create a radar chart showing personality dimensions"""
    dimensions = []
    values = []
    
    for dim, pred in predictions.items():
        dimensions.append(dim)
        values.append(pred['probability'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=dimensions,
        fill='toself',
        name='Personality Profile',
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Personality Dimensions Profile"
    )
    
    return fig

def create_confidence_bars(predictions):
    data = []
    for dim, pred in predictions.items():
        confidence = pred['probability']
        if dim == 'E/I':
            letter = 'I' if pred['prediction'] == 1 else 'E'
        elif dim == 'S/N':
            letter = 'S' if pred['prediction'] == 0 else 'N'
        elif dim == 'T/F':
            letter = 'T' if pred['prediction'] == 0 else 'F'
        elif dim == 'J/P':
            letter = 'J' if pred['prediction'] == 1 else 'P'

        data.append({
            'Dimension': f"{dim} ({letter})",
            'Confidence': confidence * 100,
            'Letter': letter
        })

    df = pd.DataFrame(data)
    fig = px.bar(df, x='Dimension', y='Confidence', title='Confidence Scores by Dimension',
                 labels={'Confidence': 'Confidence (%)', 'Dimension': 'MBTI Dimension'},
                 color='Confidence', color_continuous_scale='viridis')
    fig.update_layout(height=400)
    return fig

def main():
    st.title("🤖 MBTI Personality Predictor")
    st.markdown("Analyze text to predict Myers-Briggs personality type!")

    st.sidebar.header("📊 About MBTI")
    st.sidebar.markdown("""
    **Myers-Briggs Type Indicator** assesses personality across 4 dimensions:
    - **E/I**: Extraversion vs Introversion
    - **S/N**: Sensing vs Intuition  
    - **T/F**: Thinking vs Feeling
    - **J/P**: Judging vs Perceiving
    """)

    models, vectorizer = load_models()

    if not models:
        st.sidebar.error("❌ No models loaded")
        st.error("Ensure model files exist in the current directory")
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        st.header("📝 Enter Your Text")
        sample_texts = {
            "Analytical Person": "I enjoy breaking down complex problems and finding logical solutions in my daily work.",
            "Creative Person": "My mind is always buzzing with new ideas and imaginative ways to express myself.",
            "Social Leader": "I thrive when leading groups, motivating others, and organizing team activities for success.",
            "Thoughtful Introvert": "I prefer deep, meaningful conversations and often reflect on my thoughts and feelings alone.",
            "Empathetic Listener": "I find fulfillment in supporting friends, listening to their concerns, and offering thoughtful advice.",
            "Adventurous Explorer": "I love trying new experiences, traveling to unfamiliar places, and embracing spontaneous adventures.",
            "Organized Planner": "I keep detailed schedules, set clear goals, and enjoy creating structure in my daily life.",
            "Calm Mediator": "I help resolve conflicts by understanding different perspectives and finding peaceful solutions for everyone.",
            "Enthusiastic Motivator": "I inspire others with my positive energy and encourage them to pursue their passions fearlessly."
        }
        sample_options = ["Select a sample..."] + list(sample_texts.keys())
        selected_sample = st.selectbox("Choose a sample text:", sample_options)
        if selected_sample == "Select a sample...":
            text_value = ""
        else:
            text_value = sample_texts[selected_sample]

        text_input = st.text_area("Text to analyze (Your description):", value=text_value, height=200)
        predict_button = st.button("🔮 Predict Personality", type="primary")
        if text_input:
            word_count = len(text_input.split())
            st.info(f"📊 Word count: {word_count} words")

    with col2:
        if predict_button and text_input.strip():
            st.header("🎯 Prediction Results")
            result, error = predict_personality(text_input, models, vectorizer)
            if error:
                st.error(f"❌ {error}")
            else:
                st.markdown(f"""
                <div class="personality-result">
                    <h2>🌟 {result['personality_type']}</h2>
                    <p><strong>{result['description']}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 Dimension Analysis")
                for dim, pred in result['dimensions'].items():
                    confidence = pred['probability']
                    if dim == 'E/I':
                        letter = 'I' if pred['prediction'] == 1 else 'E'
                        
                    elif dim == 'S/N':
                        letter = 'S' if pred['prediction'] == 0 else 'N'
                        
                    elif dim == 'T/F':
                        letter = 'T' if pred['prediction'] == 0 else 'F'
                        
                    elif dim == 'J/P':
                        letter = 'J' if pred['prediction'] == 1 else 'P'
                        

                    st.markdown(f"""
                    <div class="dimension-card">
                        <strong>{dim}: {letter}</strong> ({confidence:.1%} confidence)<br>
                    </div>
                    """, unsafe_allow_html=True)
        elif predict_button:
            st.warning("⚠️ Please enter some text to analyze!")

    if predict_button and text_input.strip() and 'result' in locals() and result:
        st.header("📈 Personality Profile Visualization")
        col3, col4 = st.columns(2)
        with col3:
            radar_fig = create_personality_radar(result['dimensions'])
            st.plotly_chart(radar_fig, use_container_width=True)
        with col4:
            bar_fig = create_confidence_bars(result['dimensions'])
            st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("📋 Detailed Results")
        results_data = []
        for dim, pred in result['dimensions'].items():
            if dim == 'E/I':
                letter = 'I' if pred['prediction'] == 1 else 'E'
            elif dim == 'S/N':
                letter = 'S' if pred['prediction'] == 0 else 'N'
            elif dim == 'T/F':
                letter = 'T' if pred['prediction'] == 0 else 'F'
            elif dim == 'J/P':
                letter = 'J' if pred['prediction'] == 1 else 'P'
            results_data.append({
                'Predicted': letter,
                'Confidence': f"{pred['probability']:.1%}"
            })
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("🔍 Learn More About MBTI Types"):
        for ptype, desc in PERSONALITY_TYPES.items():
            st.write(f"**{ptype}**: {desc}")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 MBTI Personality Predictor | Built with Streamlit & Machine Learning</p>
        <p><small>Note: This is for entertainment/research purposes.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
