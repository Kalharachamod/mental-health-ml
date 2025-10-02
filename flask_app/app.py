import os
import sys
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import COLUMNS
from src.data import _to_float, _norm_category, _norm_yes_no, _clip_ranges

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Load models function
def load_models():
    base_path = os.path.join(os.path.dirname(__file__), "..", "models")
    
    print("Loading models from:", base_path)
    
    condition_model = joblib.load(os.path.join(base_path, "condition_model.pkl"))
    stress_model = joblib.load(os.path.join(base_path, "stress_model.pkl"))
    severity_model = joblib.load(os.path.join(base_path, "severity_model.pkl"))
    
    condition_encoder = joblib.load(os.path.join(base_path, "condition_label_encoder.pkl"))
    stress_encoder = joblib.load(os.path.join(base_path, "stress_encoder.pkl"))
    severity_encoder = joblib.load(os.path.join(base_path, "severity_encoder.pkl"))
    
    print("Models loaded successfully!")
    print("Condition model type:", type(condition_model))
    print("Stress model type:", type(stress_model))
    print("Severity model type:", type(severity_model))
    
    return condition_model, stress_model, severity_model, condition_encoder, stress_encoder, severity_encoder

# Load models at startup
try:
    condition_model, stress_model, severity_model, condition_encoder, stress_encoder, severity_encoder = load_models()
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    condition_model = stress_model = severity_model = None
    condition_encoder = stress_encoder = severity_encoder = None

def preprocess_input_data(data):
    """Preprocess input data to match the format expected by the models"""
    # Create a DataFrame from the input data with correct column names
    # Map the form field names to the expected column names
    mapped_data = {
        "Age": data["Age"],
        "Sleep_Hours": data["Sleep_Hours"],
        "Work_Hours": data["Work_Hours"],
        "Physical_Activity_Hours": data["Physical_Activity_Hours"],
        "Social_Media_Usage": data["Social_Media_Usage"],
        "Gender": data["Gender"],
        "Occupation": data["Occupation"],
        "Country": data["Country"],
        "Consultation_History": data["Consultation_History"],
        "Diet_Quality": data["Diet_Quality"],
        "Smoking_Habit": data["Smoking_Habit"],
        "Alcohol_Consumption": data["Alcohol_Consumption"],
        "Medication_Usage": data["Medication_Usage"],
    }
    
    # Create a DataFrame from the mapped data
    df = pd.DataFrame([mapped_data])
    
    print("Original DataFrame:")
    print(df)
    print("DataFrame columns:", df.columns.tolist())
    
    # Apply the same preprocessing as in the training pipeline
    # Convert numerics
    for col in COLUMNS.numeric: 
        if col in df: 
            df[col] = df[col].map(_to_float)
    
    # Normalize Yes/No fields
    for col in ["Consultation_History", "Medication_Usage"]:
        if col in df:
            df[col] = df[col].map(_norm_yes_no).fillna("No").astype(str)
    
    # Normalize categorical fields
    categorical_fields = ["Gender", "Occupation", "Country", "Diet_Quality", 
                         "Smoking_Habit", "Alcohol_Consumption"]
    for col in categorical_fields:
        if col in df: 
            df[col] = df[col].map(_norm_category)
    
    # Clip ranges
    df = _clip_ranges(df)
    
    print("Preprocessed DataFrame:")
    print(df)
    print("DataFrame columns:", df.columns.tolist())
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([condition_model, stress_model, severity_model, condition_encoder, stress_encoder, severity_encoder]):
        return jsonify({'error': 'Models not loaded properly'}), 500
    
    try:
        # Get form data
        data = {
            "Age": int(request.form['age']),
            "Sleep_Hours": float(request.form['sleep_hours']),
            "Work_Hours": float(request.form['work_hours']),
            "Physical_Activity_Hours": float(request.form['activity_hours']),
            "Social_Media_Usage": float(request.form['social_media']),
            "Gender": request.form['gender'],
            "Occupation": request.form['occupation'],
            "Country": request.form['country'],
            "Consultation_History": request.form['consultation'],
            "Diet_Quality": request.form['diet'],
            "Smoking_Habit": request.form['smoking'],
            "Alcohol_Consumption": request.form['alcohol'],
            "Medication_Usage": request.form['medication'],
        }
        
        print("Raw input data:", data)
        
        # Preprocess the input data
        input_df = preprocess_input_data(data)
        
        print("Preprocessed input DataFrame shape:", input_df.shape)
        print("Preprocessed input DataFrame:\n", input_df)
        
        # Make predictions
        cond_pred = condition_model.predict(input_df)[0]
        cond_probs = condition_model.predict_proba(input_df)[0]
        cond_prob = float(max(cond_probs))
        
        stress_pred = stress_model.predict(input_df)[0]
        sev_pred = severity_model.predict(input_df)[0]
        
        print("Raw predictions:")
        print("Condition prediction:", cond_pred, type(cond_pred))
        print("Condition probabilities:", cond_probs, type(cond_probs))
        print("Stress prediction:", stress_pred, type(stress_pred))
        print("Severity prediction:", sev_pred, type(sev_pred))
        
        # Get class labels
        cond_classes = condition_model.classes_.tolist()
        stress_classes = stress_encoder.classes_.tolist()
        sev_classes = severity_encoder.classes_.tolist()
        
        print("Class labels:")
        print("Condition classes:", cond_classes)
        print("Stress classes:", stress_classes)
        print("Severity classes:", sev_classes)
        
        # Decode predictions
        stress_label = stress_encoder.inverse_transform([int(stress_pred)])[0]
        severity_label = severity_encoder.inverse_transform([int(sev_pred)])[0]
        
        print("Decoded labels:")
        print("Stress label:", stress_label)
        print("Severity label:", severity_label)
        
        # Generate personalized recommendations
        recommendations = generate_recommendations(data, stress_label, severity_label)
        
        # Calculate risk scores
        risk_scores = calculate_risk_scores(data, stress_label, severity_label)
        
        # Generate wellness insights
        wellness_insights = generate_wellness_insights(data, stress_label, severity_label)
        
        # Format results
        results = {
            'condition': str(cond_pred),
            'condition_risk': risk_category(cond_prob),
            'condition_confidence': cond_prob,
            'stress_level': str(stress_label),
            'severity': str(severity_label),
            'condition_probabilities': dict(zip([str(cls) for cls in cond_classes], [float(prob) for prob in cond_probs])),
            'recommendations': recommendations,
            'risk_scores': risk_scores,
            'wellness_insights': wellness_insights,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_id': generate_analysis_id(data)
        }
        
        print("Final results:", results)
        
        return jsonify(results)
        
    except Exception as e:
        print("Error in prediction:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_risk_color(score):
    """Helper function to get color based on risk score"""
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (ValueError, TypeError):
            return colors.HexColor('#388E3C')  # Default to green
    
    if score >= 75:
        return colors.HexColor('#D32F2F')  # Red
    elif score >= 50:
        return colors.HexColor('#F57C00')  # Orange
    elif score >= 25:
        return colors.HexColor('#FBC02D')  # Yellow
    else:
        return colors.HexColor('#388E3C')  # Green

def generate_analysis_id(data):
    """Generate a unique analysis ID based on input data"""
    # Create a simple hash of the input data for tracking
    import hashlib
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()[:8]

def calculate_risk_scores(data, stress_level, severity_level):
    """Calculate detailed risk scores across multiple dimensions"""
    scores = {}
    
    # Sleep quality score (0-100)
    if data['Sleep_Hours'] < 4:
        scores['sleep'] = 90
    elif data['Sleep_Hours'] < 6:
        scores['sleep'] = 60
    elif data['Sleep_Hours'] <= 9:
        scores['sleep'] = 20
    else:
        scores['sleep'] = 40
    
    # Work stress score (0-100)
    if data['Work_Hours'] > 60:
        scores['work'] = 85
    elif data['Work_Hours'] > 50:
        scores['work'] = 65
    elif data['Work_Hours'] > 40:
        scores['work'] = 40
    else:
        scores['work'] = 20
    
    # Physical health score (0-100)
    if data['Physical_Activity_Hours'] < 1:
        scores['physical'] = 80
    elif data['Physical_Activity_Hours'] < 3:
        scores['physical'] = 50
    elif data['Physical_Activity_Hours'] < 6:
        scores['physical'] = 25
    else:
        scores['physical'] = 10
    
    # Social media impact score (0-100)
    if data['Social_Media_Usage'] > 6:
        scores['social_media'] = 75
    elif data['Social_Media_Usage'] > 3:
        scores['social_media'] = 45
    else:
        scores['social_media'] = 20
    
    # Overall mental health risk score
    base_score = (scores['sleep'] + scores['work'] + scores['physical'] + scores['social_media']) / 4
    
    # Adjust based on model predictions
    if stress_level == 'High':
        base_score += 15
    elif stress_level == 'Medium':
        base_score += 5
    
    if severity_level == 'High':
        base_score += 20
    elif severity_level == 'Medium':
        base_score += 10
    
    scores['overall'] = min(100, max(0, base_score))
    
    return scores

def generate_wellness_insights(data, stress_level, severity_level):
    """Generate deep wellness insights based on patterns in the data"""
    insights = []
    
    # Sleep pattern insight
    if data['Sleep_Hours'] < 6:
        insights.append({
            'title': 'Sleep',
            'description': 'Chronic sleep deprivation can significantly impact cognitive function and emotional regulation.',
            'impact': 'High',
            'suggestion': 'Prioritize consistent sleep schedules and create a technology-free bedtime routine.'
        })
    elif data['Sleep_Hours'] > 9:
        insights.append({
            'title': 'Sleep',
            'description': 'Excessive sleep may indicate underlying health issues or depression.',
            'impact': 'Medium',
            'suggestion': 'Monitor sleep quality, not just quantity. Consider consulting a sleep specialist.'
        })
    
    # Work-life balance insight
    if data['Work_Hours'] > 50:
        insights.append({
            'title': 'Work-Life Balance',
            'description': 'Working more than 50 hours per week significantly increases risk of burnout and mental health issues.',
            'impact': 'High',
            'suggestion': 'Implement strict work boundaries and schedule regular decompression activities.'
        })
    
    # Physical activity insight
    if data['Physical_Activity_Hours'] < 2:
        insights.append({
            'title': 'Physical Health',
            'description': 'Regular physical activity is one of the most effective natural antidepressants.',
            'impact': 'High',
            'suggestion': 'Start with 10-minute walks and gradually increase to 150 minutes of moderate exercise weekly.'
        })
    
    # Age-related insight
    if data['Age'] < 25:
        insights.append({
            'title': 'Developmental Stage',
            'description': 'Young adults face unique mental health challenges including academic pressure and identity formation.',
            'impact': 'Medium',
            'suggestion': 'Build strong social connections and develop healthy coping mechanisms for transition stress.'
        })
    elif data['Age'] > 60:
        insights.append({
            'title': 'Life Stage',
            'description': 'Older adults may experience isolation and health-related anxiety.',
            'impact': 'Medium',
            'suggestion': 'Maintain social engagement and focus on purposeful activities.'
        })
    
    # Stress-severity correlation insight
    if stress_level == 'High' and severity_level in ['High', 'Medium']:
        insights.append({
            'title': 'Risk Assessment',
            'description': 'High stress combined with significant symptoms indicates need for professional intervention.',
            'impact': 'Critical',
            'suggestion': 'Immediate consultation with mental health professionals is strongly recommended.'
        })
    
    return insights

def generate_recommendations(data, stress_level, severity_level):
    """Generate personalized recommendations based on user input and predictions"""
    recommendations = []
    
    # Sleep recommendations
    if data['Sleep_Hours'] < 6:
        recommendations.append({
            'title': 'Improve Sleep Quality',
            'description': 'Aim for 7-9 hours of sleep nightly. Try maintaining a consistent sleep schedule and creating a relaxing bedtime routine.',
            'icon': 'fas fa-bed',
            'priority': 'high',
            'category': 'Sleep'
        })
    elif data['Sleep_Hours'] > 9:
        recommendations.append({
            'title': 'Optimize Sleep Schedule',
            'description': 'While adequate sleep is important, too much can indicate issues. Aim for 7-9 hours for optimal health.',
            'icon': 'fas fa-bed',
            'priority': 'medium',
            'category': 'Sleep'
        })
    
    # Work-life balance recommendations
    if data['Work_Hours'] > 50:
        recommendations.append({
            'title': 'Work-Life Balance',
            'description': 'Consider setting boundaries between work and personal time. Take regular breaks to prevent burnout.',
            'icon': 'fas fa-briefcase',
            'priority': 'high',
            'category': 'Work'
        })
    
    # Physical activity recommendations
    if data['Physical_Activity_Hours'] < 2:
        recommendations.append({
            'title': 'Increase Physical Activity',
            'description': 'Aim for at least 150 minutes of moderate exercise per week. Even short walks can improve mental health.',
            'icon': 'fas fa-walking',
            'priority': 'high',
            'category': 'Physical'
        })
    
    # Social media recommendations
    if data['Social_Media_Usage'] > 4:
        recommendations.append({
            'title': 'Digital Wellness',
            'description': 'Limit social media usage to reduce anxiety. Try scheduling device-free periods during the day.',
            'icon': 'fas fa-mobile-alt',
            'priority': 'medium',
            'category': 'Digital'
        })
    
    # Stress-related recommendations
    if stress_level in ['High', 'Medium']:
        recommendations.append({
            'title': 'Stress Management',
            'description': 'Practice stress reduction techniques such as meditation, deep breathing, or yoga.',
            'icon': 'fas fa-spa',
            'priority': 'high',
            'category': 'Mental'
        })
    
    # Severity-related recommendations
    if severity_level in ['High', 'Medium']:
        recommendations.append({
            'title': 'Professional Support',
            'description': 'Consider consulting with a mental health professional for personalized guidance and support.',
            'icon': 'fas fa-user-md',
            'priority': 'critical',
            'category': 'Professional'
        })
    
    # Diet recommendations
    if data['Diet_Quality'] == 'Unhealthy':
        recommendations.append({
            'title': 'Nutritional Improvement',
            'description': 'Focus on a balanced diet rich in fruits, vegetables, and whole grains to support mental health.',
            'icon': 'fas fa-apple-alt',
            'priority': 'medium',
            'category': 'Nutrition'
        })
    
    # Smoking recommendations
    if data['Smoking_Habit'] in ['Regular Smoker', 'Heavy Smoker']:
        recommendations.append({
            'title': 'Smoking Cessation',
            'description': 'Consider seeking support to quit smoking, which can significantly improve both physical and mental health.',
            'icon': 'fas fa-ban',
            'priority': 'high',
            'category': 'Health'
        })
    
    # Alcohol recommendations
    if data['Alcohol_Consumption'] == 'Regular Drinker':
        recommendations.append({
            'title': 'Alcohol Moderation',
            'description': 'Consider reducing alcohol consumption, as it can negatively impact mental health and sleep quality.',
            'icon': 'fas fa-wine-bottle',
            'priority': 'medium',
            'category': 'Health'
        })
    
    # Positive reinforcement
    if len(recommendations) == 0:
        recommendations.append({
            'title': 'Maintain Healthy Habits',
            'description': 'Great job maintaining healthy lifestyle habits! Continue your positive practices to support mental wellness.',
            'icon': 'fas fa-thumbs-up',
            'priority': 'low',
            'category': 'Maintenance'
        })
    
    return recommendations

def get_risk_level_text(score):
    """Helper function to convert risk score to text description"""
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (ValueError, TypeError):
            return "N/A"
    
    if score >= 75:
        return "Very High Risk"
    elif score >= 50:
        return "High Risk"
    elif score >= 25:
        return "Moderate Risk"
    else:
        return "Low Risk"

def risk_category(prob: float) -> str:
    """Helper function to categorize risk"""
    if prob < 0.2:
        return "Very Low Risk"
    elif prob < 0.4:
        return "Low Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    elif prob < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Get the analysis results from the request
        data = request.get_json()
        
        # Create a BytesIO buffer to hold the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#8A2BE2')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#00BFFF')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#6A1B9A')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Title
        story.append(Paragraph("MindScope Analytics Mental Health Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(
            f"This report provides a comprehensive analysis of your mental health status based on the data you provided. "
            f"The analysis was completed on {data.get('timestamp', 'N/A')} with Analysis ID: {data.get('analysis_id', 'N/A')}.",
            normal_style
        ))
        story.append(Spacer(1, 10))
        
        # Key Findings
        story.append(Paragraph("Key Findings", heading_style))
        story.append(Paragraph(
            f"<b>Condition:</b> {data.get('condition', 'N/A')}<br/>"
            f"<b>Stress Level:</b> {data.get('stress_level', 'N/A')}<br/>"
            f"<b>Severity:</b> {data.get('severity', 'N/A')}<br/>"
            f"<b>Condition Risk:</b> {data.get('condition_risk', 'N/A')}<br/>"
            f"<b>Confidence:</b> {round(data.get('condition_confidence', 0) * 100, 1) if data.get('condition_confidence') else 'N/A'}%",
            normal_style
        ))
        story.append(Spacer(1, 20))
        
        # Deep Wellness Insights Section
        story.append(Paragraph("Deep Wellness Insights", heading_style))
        story.append(Paragraph(
            "The following insights provide a detailed analysis of factors that may be affecting your mental wellness:",
            normal_style
        ))
        story.append(Spacer(1, 10))
        
        # Add wellness insights
        wellness_insights = data.get('wellness_insights', [])
        if wellness_insights:
            for insight in wellness_insights:
                story.append(Paragraph(f"<b>{insight.get('title', 'N/A')}</b>", subheading_style))
                story.append(Paragraph(
                    f"<b>Description:</b> {insight.get('description', 'N/A')}<br/>"
                    f"<b>Suggestion:</b> {insight.get('suggestion', 'N/A')}",
                    normal_style
                ))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No wellness insights available.", normal_style))
        
        story.append(PageBreak())
        
        # Personalized Action Plan Section
        story.append(Paragraph("Personalized Action Plan", heading_style))
        story.append(Paragraph(
            "Based on your profile and assessment, here are personalized recommendations to improve your mental health:",
            normal_style
        ))
        story.append(Spacer(1, 10))
        
        # Add recommendations
        recommendations = data.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                priority = rec.get('priority', 'N/A')
                priority_color = colors.red if priority == 'Critical' else \
                                colors.orange if priority == 'High' else \
                                colors.yellow if priority == 'Medium' else \
                                colors.green
                
                story.append(Paragraph(
                    f"<b>{rec.get('title', 'N/A')}</b> "
                    f"<font color='{priority_color.hexval()}'>[{priority}]</font>",
                    subheading_style
                ))
                story.append(Paragraph(rec.get('description', 'N/A'), normal_style))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No recommendations available.", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Disclaimer", heading_style))
        story.append(Paragraph(
            "This report is generated by an AI system and is intended for informational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of your physician or qualified health provider with any questions "
            "you may have regarding a medical condition.",
            normal_style
        ))
        
        # Build the PDF
        doc.build(story)
        
        # Move to the beginning of the buffer
        buffer.seek(0)
        
        # Return the PDF as a response
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"mental_health_report_{data.get('analysis_id', 'unknown')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate report'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)