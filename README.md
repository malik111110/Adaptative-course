# Adaptive Courses API

The Adaptive Courses API is a Flask-based backend designed to provide personalized course recommendations and dropout risk analysis for students in an online learning platform. Built with Python, it leverages machine learning (Random Forest classifier) and collaborative filtering to tailor recommendations based on student profiles, learning styles, and peer performance. The API supports three main endpoints: `/analysis`, `/students/<id>`, and `/recommendations/<id>`.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Algorithms and AI Classifier](#algorithms-and-ai-classifier)
  - [Dropout Prediction (Random Forest Classifier)](#dropout-prediction-random-forest-classifier)
  - [Recommendation Algorithm](#recommendation-algorithm)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Dropout Risk Prediction**: Uses a Random Forest classifier to predict dropout likelihood based on student features.
- **Personalized Recommendations**: Combines content-based filtering (learning style match) and collaborative filtering (similar student success) with a dropout risk adjustment.
- **Analytics Dashboard Support**: Provides aggregated statistics for visualizing dropout risk, engagement, and course performance.
- **Scalable Design**: Modular structure with separate services for data management, preprocessing, and recommendations.

---

## Project Structure
```
Adaptative-courses/
├── personalized_learning_dataset.csv  # Dataset with student data
├── recommender/                       # Core application code
│   ├── __init__.py
│   ├── api/                          # Flask API layer
│   │   └── app.py
│   ├── algorithms/                   # Recommendation and similarity algorithms
│   │   ├── recommender.py
│   │   └── similarity.py
│   ├── ai/                           # AI components (classifier and preprocessor)
│   │   ├── classifier.py
│   │   └── preprocessor.py
│   ├── core/                         # Core models and services
│   │   ├── models.py
│   │   └── services.py
│   └── data/                         # Data management
│       └── manager.py
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## Dataset
The API uses `personalized_learning_dataset.csv`, a synthetic dataset with 10,000 student records. Each row represents a student’s interaction with a course.

### Columns
- **Student_ID**: Unique identifier (e.g., `S00001`).
- **Age**: Integer (18-40 range).
- **Gender**: `Male`, `Female`, or `Other`.
- **Education_Level**: `High School`, `Bachelor's`, `Master's`, or `PhD`.
- **Learning_Style**: `Visual`, `Auditory`, `Reading/Writing`, or `Kinesthetic`.
- **Course_Name**: One of `Python Basics`, `Web Development`, `Data Science`, `Machine Learning`, or `Cybersecurity`.
- **Time_Spent_on_Videos**: Float (hours spent on video content).
- **Quiz_Scores**: Float (average quiz score, 0-100).
- **Quiz_Attempts**: Integer (number of quiz attempts).
- **Forum_Participation**: Float (participation score, 0-100).
- **Assignment_Completion_Rate**: Float (percentage, 0-100).
- **Final_Exam_Score**: Float (final score, 0-100).
- **Feedback_Score**: Integer (student feedback, 1-5).
- **Engagement_Level**: `Low`, `Medium`, or `High`.
- **Dropout_Likelihood**: `Yes` or `No` (target variable for classifier).

### Statistics
- **Size**: 10,000 rows.
- **Dropout Rate**: ~5.7% (based on sample, ~570 `Yes`).
- **Courses**: 5 unique courses, roughly balanced (~1,953-2,043 students each).

---

## Algorithms and AI Classifier

### Dropout Prediction (Random Forest Classifier)
The API uses a Random Forest classifier to predict dropout risk, implemented in `recommender/ai/classifier.py`.

#### Features
- **Input Features** (preprocessed in `recommender/ai/preprocessor.py`):
  - Age (normalized).
  - Gender (one-hot encoded: Male, Female, Other).
  - Education Level (one-hot encoded: High School, Bachelor's, Master's, PhD).
  - Learning Style (one-hot encoded: Visual, Auditory, Reading/Writing, Kinesthetic).
  - Engagement Metrics (per course): Time Spent on Videos, Quiz Scores, Forum Participation, Assignment Completion Rate (normalized).
  - Engagement Level (one-hot encoded: Low, Medium, High).
- **Target**: `Dropout_Likelihood` (`Yes` → 1, `No` → 0).

#### Model Details
- **Algorithm**: Random Forest (`sklearn.ensemble.RandomForestClassifier`).
- **Parameters**:
  - `n_estimators=100`: 100 decision trees.
  - `class_weight="balanced"`: Adjusts for class imbalance (~5.7% dropouts).
  - `random_state=42`: Ensures reproducibility.
- **Training**:
  - 80/20 train-test split.
  - Evaluated with a custom threshold of 0.3 for binary prediction (though stored scores are probabilities).
- **Output**: `predicted_dropout_score` (0-1 probability) stored in each `StudentProfile`.

#### Performance (Threshold 0.3)
- **Accuracy**: 78.45%.
- **Classification Report**:
  ```
  precision    recall  f1-score   support
  No Dropout    0.81      0.96      0.88      1623
  Dropout       0.20      0.05      0.08       377
  ```
- **Notes**: Low recall (0.05) for dropouts due to conservative threshold and imbalance. Scores like 0.7 (e.g., `S00027`) correctly identify high-risk students.

### Recommendation Algorithm
The recommendation system, implemented in `recommender/algorithms/recommender.py`, combines content-based and collaborative filtering with a dropout risk adjustment.

#### Components
1. **Content-Based Filtering**:
   - Matches student’s `learning_style` to course `content_type_weights` (defined in `services.py`).
   - Example Weights:
     - Python Basics: Visual (0.3), Auditory (0.1), Reading/Writing (0.5), Kinesthetic (0.1).
     - Web Development: Visual (0.4), Auditory (0.1), Reading/Writing (0.2), Kinesthetic (0.3).
   - Score: `content_match = course.content_type_weights[student.learning_style]`.

2. **Collaborative Filtering**:
   - Uses cosine similarity (`recommender/algorithms/similarity.py`) to find 5 most similar students based on profile features (age, gender, education, etc.).
   - Calculates average success (`final_exam_scores / 100`) among similar students who took the course.
   - Score: `collab_score = avg(similar_students_success)` (0 if no similar students took the course).

3. **Dropout Risk Adjustment**:
   - If `predicted_dropout_score > 0.5`, adds a fixed `+0.25` to the relevance score.
   - Encourages courses for high-risk students to improve retention.

#### Formula
```
relevance_score = 0.5 * content_match + 0.5 * collab_score + (0.25 if predicted_dropout_score > 0.5 else 0)
```

#### Reasoning
- Combines: "Matches learning style (Visual: X.XX)", "Popular among N similar students (avg success: X.XX)", and "Adjusted for high dropout risk (+0.25)" (if applicable).

#### Example (`S00027`)
- **Profile**: Visual learner, `predicted_dropout_score=0.7`.
- **Web Development**:
  - Content Match: 0.40.
  - Collab Score: 0.84 (3 similar students).
  - Adjustment: +0.25.
  - Score: `0.5 * 0.40 + 0.5 * 0.84 + 0.25 = 0.87`.

---

## Setup and Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Adaptative-courses
   ```
   Or copy the project folder to your machine.

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Dataset**:
   - Ensure `personalized_learning_dataset.csv` is in the root directory (`/Users/mac/Desktop/Adaptative-courses/`).

---

## Usage

### Running the API
```bash
cd /Users/mac/Desktop/Adaptative-courses/
python -m recommender.api.app
```
- Runs on `http://127.0.0.1:5000` (and all network interfaces).
- Debug mode is enabled by default.

### API Endpoints
1. **GET `/analysis`**:
   - **Description**: Returns aggregated statistics for dropout risk, engagement, and course performance.
   - **Response**:
     ```json
     {
       "avg_dropout_risk": 0.1976,
       "course_statistics": {...},
       "dropout_risk_distribution": {"0-0.25": 9995, "0.25-0.5": 5, ...},
       "engagement_distribution": {"High": 2980, "Medium": 4927, "Low": 2093},
       "total_students": 10000
     }
     ```

2. **GET `/students/<student_id>`**:
   - **Description**: Returns a student’s profile.
   - **Example**: `curl http://localhost:5000/students/S00027`
   - **Response**:
     ```json
     {
       "student_id": "S00027",
       "age": 30,
       "gender": "Male",
       "education_level": "High School",
       "learning_style": "Visual",
       "course_history": ["Python Basics"],
       "engagement_level": "Medium",
       "dropout_likelihood": true,
       "predicted_dropout_score": 0.7
     }
     ```

3. **GET `/recommendations/<student_id>`**:
   - **Description**: Returns top 3 course recommendations.
   - **Example**: `curl http://localhost:5000/recommendations/S00027`
   - **Response**:
     ```json
     {
       "student_id": "S00027",
       "recommendations": [
         {
           "course_name": "Web Development",
           "reasoning": "Matches learning style (Visual: 0.40). Popular among 3 similar students (avg success: 0.84). Adjusted for high dropout risk (+0.25).",
           "relevance_score": 0.87
         },
         {...}
       ]
     }
     ```

---

## Dependencies
Listed in `requirements.txt`:
```
flask==2.3.3
scikit-learn==1.5.0
numpy==1.26.4
```
Install with:
```bash
pip install -r requirements.txt
```

---

## Future Improvements
- **Classifier Tuning**: Increase dropout recall by lowering the threshold (e.g., 0.2) or adding features (e.g., quiz attempts).
- **Dynamic Weights**: Adjust `content_type_weights` based on student feedback or course outcomes.
- **Scalability**: Add database support (e.g., PostgreSQL) instead of in-memory storage.
- **API Security**: Implement authentication and rate limiting for production use.
- **Real-Time Updates**: Support live data ingestion for continuous learning.

---

## Contributing
- Fork the repository, make changes, and submit a pull request.
- Report issues or suggest features via GitHub Issues (if hosted).

---

## License
This project is unlicensed for now—add an appropriate license (e.g., MIT) based on your needs.

---

### Notes on the Update
- **Table of Contents**: Now uses proper Markdown syntax (`- [Section](#section)`), linking to each header with lowercase and hyphenated anchors (e.g., `#algorithms-and-ai-classifier`).
- **Content**: Remains unchanged from the previous version, just with the corrected TOC.