"""
Flask Application for MBTI Personality Prediction

This application provides two prediction methods:
1. Text-based prediction (using ML models)
2. MCQ-based prediction (using scoring engine)

Both methods coexist and can be accessed via different routes.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from mcq_scoring import MCQScoringEngine

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# MBTI personality type descriptions
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

# Global variables for models (loaded once at startup)
models = {}
vectorizer = None
mcq_engine = None


def load_ml_models():
    """
    Load ML models and vectorizer for text-based prediction.
    This function is called once at application startup.
    """
    global models, vectorizer
    
    model_files = {
        'E/I': 'mbti_model_ie.joblib',
        'S/N': 'mbti_model_ns.joblib',
        'T/F': 'mbti_model_ft.joblib',
        'J/P': 'mbti_model_jp.joblib'
    }
    
    vectorizer_file = 'mbti_vectorizer.joblib'
    
    # Load vectorizer
    if os.path.exists(vectorizer_file):
        try:
            vectorizer = joblib.load(vectorizer_file)
        except Exception as e:
            print(f"Warning: Could not load vectorizer: {e}")
    
    # Load models
    for dimension, filename in model_files.items():
        if os.path.exists(filename):
            try:
                models[dimension] = joblib.load(filename)
            except Exception as e:
                print(f"Warning: Could not load model {filename}: {e}")


def load_mcq_engine():
    """
    Load MCQ scoring engine.
    This function is called once at application startup.
    """
    global mcq_engine
    try:
        mcq_engine = MCQScoringEngine('mcq_questions.json')
    except Exception as e:
        print(f"Warning: Could not load MCQ engine: {e}")


def predict_personality_from_text(text):
    """
    Predict MBTI personality type from text input using ML models.
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        tuple: (result_dict, error_message)
            result_dict contains:
                - personality_type: str
                - dimensions: dict
                - description: str
    """
    if not models or not vectorizer:
        return None, "Models not loaded. Please ensure model files exist."
    
    try:
        # Transform text to features
        text_features = vectorizer.transform([text])
        predictions = {}
        personality_type = ""
        
        # Predict each dimension
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
                
                # Build personality type string
                if dimension == 'E/I':
                    personality_type += 'I' if predictions[dimension]['prediction'] == 1 else 'E'
                elif dimension == 'S/N':
                    personality_type += 'S' if predictions[dimension]['prediction'] == 0 else 'N'
                elif dimension == 'T/F':
                    personality_type += 'T' if predictions[dimension]['prediction'] == 0 else 'F'
                elif dimension == 'J/P':
                    personality_type += 'J' if predictions[dimension]['prediction'] == 1 else 'P'
                    
            except Exception as e:
                predictions[dimension] = {'probability': 0.5, 'prediction': 0}
                personality_type += 'X'
        
        return {
            'personality_type': personality_type,
            'dimensions': predictions,
            'description': PERSONALITY_TYPES.get(personality_type, 'Unknown personality type')
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


def load_mbti_descriptions():
    """
    Load detailed MBTI descriptions from JSON for the homepage modal.
    Returns a dict keyed by MBTI type, or empty dict if file missing.
    """
    path = os.path.join(os.path.dirname(__file__), 'mbti_descriptions.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


@app.route('/')
def index():
    """Home page with navigation to both prediction methods and 16 type cards with More modals."""
    mbti_descriptions = load_mbti_descriptions()
    return render_template(
        'index.html',
        personality_types=PERSONALITY_TYPES,
        mbti_descriptions=mbti_descriptions
    )


@app.route('/text', methods=['GET', 'POST'])
def text_prediction():
    """
    Text-based MBTI prediction route.
    Handles both GET (display form) and POST (process prediction).
    """
    if request.method == 'POST':
        text_input = request.form.get('text_input', '').strip()
        
        if not text_input:
            return render_template('text_prediction.html', 
                                 error="Please enter some text to analyze.")
        
        result, error = predict_personality_from_text(text_input)
        
        if error:
            return render_template('text_prediction.html', error=error)
        
        return render_template('text_prediction.html', 
                             result=result, 
                             input_text=text_input)
    
    # GET request - show form
    return render_template('text_prediction.html')


@app.route('/mcq', methods=['GET', 'POST'])
def mcq_prediction():
    """
    MCQ-based MBTI prediction route.
    Handles both GET (display questions) and POST (process responses).
    """
    if not mcq_engine:
        return render_template('mcq_prediction.html', 
                             error="MCQ engine not loaded. Please check mcq_questions.json file.")
    
    if request.method == 'POST':
        # Collect responses from form
        responses = {}
        for key, value in request.form.items():
            if key.startswith('question_'):
                question_id = key.replace('question_', '')
                responses[question_id] = value
        
        # Validate responses
        is_valid, missing = mcq_engine.validate_responses(responses)
        
        if not is_valid:
            questions = mcq_engine.get_all_questions()
            return render_template('mcq_prediction.html',
                                 questions=questions,
                                 error=f"Please answer all questions. Missing: {len(missing)} question(s).",
                                 responses=responses)
        
        # Calculate scores
        scores = mcq_engine.calculate_scores(responses)
        
        # Predict MBTI type
        mbti_type = mcq_engine.predict_mbti(scores)
        
        # Calculate confidence
        confidence = mcq_engine.calculate_confidence(scores)
        
        # Get trait for each dimension
        dimension_results = {}
        dimension_mapping = {
            'EI': 'EI_score',
            'SN': 'SN_score',
            'TF': 'TF_score',
            'JP': 'JP_score'
        }
        
        for dimension, score_key in dimension_mapping.items():
            score = scores[score_key]
            trait = mcq_engine.get_dimension_trait(dimension, score)
            dimension_results[dimension] = {
                'trait': trait,
                'score': score,
                'confidence': confidence[dimension]
            }
        
        result = {
            'personality_type': mbti_type,
            'description': PERSONALITY_TYPES.get(mbti_type, 'Unknown personality type'),
            'scores': scores,
            'confidence': confidence,
            'dimensions': dimension_results
        }
        
        return render_template('mcq_results.html', result=result)
    
    # GET request - show questions
    questions = mcq_engine.get_all_questions()
    return render_template('mcq_prediction.html', questions=questions)


@app.route('/about')
def about():
    """About page explaining MBTI and the prediction methods."""
    return render_template('about.html', personality_types=PERSONALITY_TYPES)


# Initialize models and engine at startup
@app.before_request
def before_request():
    """Ensure models are loaded before handling requests."""
    global models, vectorizer, mcq_engine
    if not models:
        load_ml_models()
    if not mcq_engine:
        load_mcq_engine()


if __name__ == '__main__':
    # Load models at startup
    load_ml_models()
    load_mcq_engine()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
