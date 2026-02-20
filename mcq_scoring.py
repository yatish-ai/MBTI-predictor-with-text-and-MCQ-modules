"""
MCQ Scoring Engine Module

This module handles scoring logic for MCQ-based MBTI personality prediction.
It processes user responses and calculates scores for each MBTI dimension.
"""

import json
import os


class MCQScoringEngine:
    """
    Scoring engine for MCQ-based MBTI personality prediction.
    
    Maintains four independent scores:
    - EI_score: Extraversion (positive) vs Introversion (negative)
    - SN_score: Sensing (positive) vs Intuition (negative)
    - TF_score: Thinking (positive) vs Feeling (negative)
    - JP_score: Judging (positive) vs Perceiving (negative)
    """
    
    def __init__(self, questions_file='mcq_questions.json'):
        """
        Initialize the scoring engine.
        
        Args:
            questions_file (str): Path to JSON file containing MCQ questions
        """
        self.questions_file = questions_file
        self.questions = self._load_questions()
        self.dimension_counts = self._count_questions_per_dimension()
    
    def _load_questions(self):
        """
        Load questions from JSON file.
        
        Returns:
            list: List of question dictionaries
        """
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('questions', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Questions file '{self.questions_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in '{self.questions_file}'.")
    
    def _count_questions_per_dimension(self):
        """
        Count total questions for each MBTI dimension.
        
        Returns:
            dict: Dictionary mapping dimension to question count
        """
        counts = {'EI': 0, 'SN': 0, 'TF': 0, 'JP': 0}
        for question in self.questions:
            dimension = question.get('dimension', '')
            if dimension in counts:
                counts[dimension] += 1
        return counts
    
    def calculate_scores(self, responses):
        """
        Calculate scores for each MBTI dimension based on user responses.
        
        Args:
            responses (dict): Dictionary mapping question_id to selected_option (A, B, C, or D)
                            Example: {1: 'A', 2: 'B', ...}
        
        Returns:
            dict: Dictionary containing scores for each dimension
                {
                    'EI_score': int,
                    'SN_score': int,
                    'TF_score': int,
                    'JP_score': int
                }
        """
        # Initialize scores
        scores = {
            'EI_score': 0,
            'SN_score': 0,
            'TF_score': 0,
            'JP_score': 0
        }
        
        # Process each response
        for question in self.questions:
            question_id = question.get('id')
            dimension = question.get('dimension', '')
            selected_option = responses.get(str(question_id)) or responses.get(question_id)
            
            if selected_option and selected_option in question.get('options', {}):
                option_data = question['options'][selected_option]
                score_value = option_data.get('score', 0)
                
                # Update the appropriate dimension score
                if dimension == 'EI':
                    scores['EI_score'] += score_value
                elif dimension == 'SN':
                    scores['SN_score'] += score_value
                elif dimension == 'TF':
                    scores['TF_score'] += score_value
                elif dimension == 'JP':
                    scores['JP_score'] += score_value
        
        return scores
    
    def predict_mbti(self, scores):
        """
        Predict MBTI personality type based on scores.
        
        Prediction rules:
        - EI_score > 0 → E else I
        - SN_score > 0 → S else N
        - TF_score > 0 → T else F
        - JP_score > 0 → J else P
        
        Args:
            scores (dict): Dictionary containing dimension scores
        
        Returns:
            str: Four-letter MBTI personality type (e.g., 'INTJ', 'ENFP')
        """
        mbti_type = ""
        
        # Determine each dimension based on score
        mbti_type += 'E' if scores['EI_score'] > 0 else 'I'
        mbti_type += 'S' if scores['SN_score'] > 0 else 'N'
        mbti_type += 'T' if scores['TF_score'] > 0 else 'F'
        mbti_type += 'J' if scores['JP_score'] > 0 else 'P'
        
        return mbti_type
    
    def calculate_confidence(self, scores):
        """
        Calculate confidence for each dimension.
        
        Confidence formula: |score| / total_questions_for_dimension
        
        Args:
            scores (dict): Dictionary containing dimension scores
        
        Returns:
            dict: Dictionary mapping dimension to confidence percentage
                {
                    'EI': float (0.0 to 1.0),
                    'SN': float (0.0 to 1.0),
                    'TF': float (0.0 to 1.0),
                    'JP': float (0.0 to 1.0)
                }
        """
        confidence = {}
        
        dimension_mapping = {
            'EI': 'EI_score',
            'SN': 'SN_score',
            'TF': 'TF_score',
            'JP': 'JP_score'
        }
        
        for dimension, score_key in dimension_mapping.items():
            score = scores[score_key]
            total_questions = self.dimension_counts.get(dimension, 1)
            
            # Calculate confidence: absolute score divided by total questions
            # Normalize to range [0, 1]
            if total_questions > 0:
                confidence[dimension] = abs(score) / total_questions
            else:
                confidence[dimension] = 0.0
        
        return confidence
    
    def get_dimension_trait(self, dimension, score):
        """
        Get the predicted trait for a dimension based on score.
        
        Args:
            dimension (str): MBTI dimension ('EI', 'SN', 'TF', or 'JP')
            score (int): Score for the dimension
        
        Returns:
            str: Predicted trait (e.g., 'E', 'I', 'S', 'N', etc.)
        """
        if dimension == 'EI':
            return 'E' if score > 0 else 'I'
        elif dimension == 'SN':
            return 'S' if score > 0 else 'N'
        elif dimension == 'TF':
            return 'T' if score > 0 else 'F'
        elif dimension == 'JP':
            return 'J' if score > 0 else 'P'
        return 'X'
    
    def get_all_questions(self):
        """
        Get all questions for rendering.
        
        Returns:
            list: List of question dictionaries
        """
        return self.questions
    
    def validate_responses(self, responses):
        """
        Validate that all required questions have been answered.
        
        Args:
            responses (dict): Dictionary mapping question_id to selected_option
        
        Returns:
            tuple: (is_valid: bool, missing_questions: list)
        """
        answered_ids = set(str(k) for k in responses.keys())
        required_ids = set(str(q['id']) for q in self.questions)
        
        missing = list(required_ids - answered_ids)
        is_valid = len(missing) == 0
        
        return is_valid, missing
