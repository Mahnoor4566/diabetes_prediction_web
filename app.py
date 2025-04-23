import random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from models import db, User
import joblib
import numpy as np
import json
from flask_wtf.csrf import CSRFProtect
import os
import re

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diabetes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
csrf = CSRFProtect(app)

# Global variables for models
model_data = None
initial_model = None
models_loaded = False


# Load models before first request
@app.before_request
def load_models_on_first_request():
    global models_loaded, model_data, initial_model
    if not models_loaded and not app.testing:
        try:
            initial_model = joblib.load('models/random_forest_diabetes_model.pkl')
            model_data = joblib.load('models/web_diabetes_model.pkl')
            if not all(key in model_data for key in ['model', 'scaler', 'feature_order']):
                raise RuntimeError("Invalid model package structure")
            models_loaded = True
        except Exception as e:
            app.logger.error(f"Failed to load models: {str(e)}")
            models_loaded = False


# Load intents
with open('diabetes.json') as file:
    DIABETES_INTENTS = json.load(file)['intents']


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =====================
# AUTHENTICATION ROUTES
# =====================
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('signup'))


@app.route("/home")
@login_required
def home():
    return render_template("home.html", username=current_user.username)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash("Username or email already exists", "danger")
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            flash("Account created successfully!", "success")
            return redirect(url_for('home'))
        except Exception as e:
            db.session.rollback()
            flash("Error creating account. Please try again.", "danger")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('signup'))


# =====================
# CORE APPLICATION ROUTES
# =====================
@app.route("/learn")
@login_required
def education():
    return render_template("educational.html")


# =====================
# PREDICTION ROUTES
# =====================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        if 'gender' in request.form:
            session['gender'] = request.form.get('gender')
            return redirect(url_for('prediction_form'))
        return redirect(url_for('predict'))

    session.pop('gender', None)
    return render_template("gender.html")


@app.route("/prediction-form", methods=["GET", "POST"])
@login_required
def prediction_form():
    if 'gender' not in session:
        return redirect(url_for('predict'))

    gender = session['gender']

    if request.method == "POST":
        try:
            # Process form data
            pregnancies = int(request.form.get("Pregnancies", 0)) if gender == 'female' else 0
            input_data = [
                pregnancies,
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                int(request.form["Age"])
            ]

            # Make prediction
            input_array = np.array(input_data).reshape(1, -1)
            prediction = initial_model.predict(input_array)[0]
            proba = initial_model.predict_proba(input_array)[0][1]

            # Prepare result
            result = {
                'has_risk': prediction == 1,
                'confidence': f"{proba * 100:.1f}%",
                'additional_info': {
                    'glucose': float(request.form["Glucose"]),
                    'bmi': float(request.form["BMI"]),
                    'age': int(request.form["Age"])
                }
            }

            return render_template("predict.html",
                                   gender_selected=True,
                                   result=result,
                                   form_data=request.form)

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return render_template("predict.html",
                                   gender_selected=True,
                                   form_data=request.form)

    return render_template("predict.html", gender_selected=True)


@app.route("/detailed-prediction", methods=["GET", "POST"])
@login_required
def detailed_prediction():
    if not model_data:
        flash("Prediction system is currently unavailable", "danger")
        return redirect(url_for('predict'))

    if request.method == "POST":
        try:
            # Validate and collect form data
            form_values = {
                'HighBP': int(request.form.get('HighBP', 0)),
                'HighChol': int(request.form.get('HighChol', 0)),
                'BMI': float(request.form['BMI']),
                'HeartDiseaseorAttack': int(request.form.get('HeartDiseaseorAttack', 0)),
                'GenHlth': int(request.form['GenHlth']),
                'PhysHlth': int(request.form['PhysHlth']),
                'Age': int(request.form['Age']),
                'DiffWalk': int(request.form.get('DiffWalk', 0))
            }

            # Validate input ranges
            if not 10 <= form_values['BMI'] <= 50:
                raise ValueError("BMI must be between 10-50")
            if form_values['GenHlth'] not in [1, 2, 3, 4, 5]:
                raise ValueError("Invalid general health selection")

            # Create properly ordered DataFrame
            input_df = pd.DataFrame([form_values], columns=model_data['feature_order'])

            # Scale features and predict
            scaled_input = model_data['scaler'].transform(input_df)
            prediction = model_data['model'].predict(scaled_input)[0]
            probabilities = model_data['model'].predict_proba(scaled_input)[0]

            # Prepare results
            diagnosis_map = {
                0: 'No Diabetes',
                1: 'Pre-Diabetes',
                2: 'Type 1 Diabetes',
                3: 'Type 2 Diabetes'
            }

            recommendations = {
                0: "Maintain your healthy lifestyle with these tips:\n"
                   "• Continue regular exercise\n"
                   "• Eat balanced meals\n"
                   "• Get annual checkups",
                1: "Take action to prevent progression:\n"
                   "• Lose 5-7% body weight if overweight\n"
                   "• Increase physical activity\n"
                   "• Reduce sugar intake",
                2: "Immediate next steps:\n"
                   "• Consult endocrinologist immediately\n"
                   "• Learn insulin management\n"
                   "• Monitor blood sugar regularly",
                3: "Management recommendations:\n"
                   "• Lifestyle modification program\n"
                   "• Regular HbA1c tests\n"
                   "• Cardiovascular checkups"
            }

            return render_template('detailed_prediction.html',
                                   result={
                                       'diagnosis': diagnosis_map[prediction],
                                       'recommendations': recommendations[prediction]
                                   }
                                   )

        except Exception as e:
            flash(f"Validation Error: {str(e)}", "danger")
            return render_template('detailed_prediction.html')

    return render_template('detailed_prediction.html')


# =====================
# CHATBOT ROUTES
# =====================
@app.route("/chatbot")
@login_required
def chatbot_page():
    return render_template("chatbot.html")


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message').lower()
    response = None
    suggestions = []

    for intent in DIABETES_INTENTS:
        if any(pattern.lower() in user_message for pattern in intent['patterns']):
            response = random.choice(intent['responses'])
            suggestions = intent.get('suggestions', [])[:4]
            break

    if not response:
        fallback = next((i for i in DIABETES_INTENTS if i['tag'] == 'fallback'), None)
        if fallback:
            response = random.choice(fallback['responses'])
            suggestions = fallback.get('suggestions', [])[:4]
        else:
            response = "I'm still learning about diabetes. Could you rephrase your question?"
            suggestions = ["Diabetes symptoms?", "Prevention tips?", "Normal blood sugar levels?"]

    return jsonify({'reply': response, 'suggestions': suggestions})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=False)