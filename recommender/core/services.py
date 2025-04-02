from typing import List, Optional, Dict
from ..core.models import StudentProfile, Course, Recommendation, LearningStyle, EngagementLevel, Gender, EducationLevel
from ..algorithms.similarity import CosineSimilarity
from ..algorithms.recommender import CourseRecommender
from ..ai.preprocessor import DataPreprocessor
from ..ai.classifier import DropoutClassifier
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RecommendationService:
    def __init__(self):
        self.similarity_calculator = CosineSimilarity()
        self.recommender = CourseRecommender()
        self.students: Dict[str, StudentProfile] = {}
        self.courses: Dict[str, Course] = {}
        self.preprocessor = DataPreprocessor()
        self.classifier = DropoutClassifier()

    def load_student_from_csv_row(self, row: Dict) -> None:
        student_id = row["Student_ID"]
        course_name = row["Course_Name"]
        engagement_metrics = {
            "time_spent_on_videos": float(row["Time_Spent_on_Videos"]),
            "quiz_scores": float(row["Quiz_Scores"]),
            "forum_participation": float(row["Forum_Participation"]),
            "assignment_completion_rate": float(row["Assignment_Completion_Rate"])
        }

        if student_id in self.students:
            student = self.students[student_id]
            student.course_history.append(course_name)
            student.engagement_metrics[course_name] = engagement_metrics
            student.quiz_attempts[course_name] = int(row["Quiz_Attempts"])
            student.final_exam_scores[course_name] = float(row["Final_Exam_Score"])
            student.feedback_scores[course_name] = int(row["Feedback_Score"])
            student.last_updated = datetime.now()
        else:
            student = StudentProfile(
                student_id=student_id,
                age=int(row["Age"]),
                gender=row["Gender"],
                education_level=row["Education_Level"],
                learning_style=row["Learning_Style"],
                course_history=[course_name],
                engagement_metrics={course_name: engagement_metrics},
                quiz_attempts={course_name: int(row["Quiz_Attempts"])},
                engagement_level=row["Engagement_Level"],
                final_exam_scores={course_name: float(row["Final_Exam_Score"])},
                feedback_scores={course_name: int(row["Feedback_Score"])},
                dropout_likelihood=row["Dropout_Likelihood"] == "Yes",
                last_updated=datetime.now()
            )
            self.students[student_id] = student

        self._update_course_from_row(row)

    def _update_course_from_row(self, row: Dict) -> None:
        course_name = row["Course_Name"]
        if course_name not in self.courses:
            if course_name == "Machine Learning":
                content_type_weights = {
                    LearningStyle.VISUAL: 0.5,
                    LearningStyle.AUDITORY: 0.1,
                    LearningStyle.READING_WRITING: 0.3,
                    LearningStyle.KINESTHETIC: 0.1
                }
            elif course_name == "Python Basics":
                content_type_weights = {
                    LearningStyle.VISUAL: 0.3,
                    LearningStyle.AUDITORY: 0.1,
                    LearningStyle.READING_WRITING: 0.5,
                    LearningStyle.KINESTHETIC: 0.1
                }
            elif course_name == "Data Science":
                content_type_weights = {
                    LearningStyle.VISUAL: 0.6,
                    LearningStyle.AUDITORY: 0.1,
                    LearningStyle.READING_WRITING: 0.2,
                    LearningStyle.KINESTHETIC: 0.1
                }
            elif course_name == "Web Development":
                content_type_weights = {
                    LearningStyle.VISUAL: 0.4,
                    LearningStyle.AUDITORY: 0.1,
                    LearningStyle.READING_WRITING: 0.2,
                    LearningStyle.KINESTHETIC: 0.3
                }
            elif course_name == "Cybersecurity":
                content_type_weights = {
                    LearningStyle.VISUAL: 0.3,
                    LearningStyle.AUDITORY: 0.1,
                    LearningStyle.READING_WRITING: 0.3,
                    LearningStyle.KINESTHETIC: 0.3
                }
            else:
                content_type_weights = {style: 0.25 for style in LearningStyle}

            self.courses[course_name] = Course(
                course_name=course_name,
                content_type_weights=content_type_weights,
                average_completion_rate=float(row["Assignment_Completion_Rate"]),
                average_quiz_score=float(row["Quiz_Scores"]),
                average_time_spent=float(row["Time_Spent_on_Videos"])
            )
        else:
            course = self.courses[course_name]
            n = len([s for s in self.students.values() if course_name in s.course_history])
            course.average_completion_rate = (
                (course.average_completion_rate * (n - 1) + float(row["Assignment_Completion_Rate"])) / n
            )
            course.average_quiz_score = (
                (course.average_quiz_score * (n - 1) + float(row["Quiz_Scores"])) / n
            )
            course.average_time_spent = (
                (course.average_time_spent * (n - 1) + float(row["Time_Spent_on_Videos"])) / n
            )

    def train_classifier(self) -> None:
        """Train a Random Forest classifier and apply a custom threshold."""
        students = list(self.students.values())
        X = self.preprocessor.preprocess_students(students)
        y = [student.dropout_likelihood for student in students]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train with Random Forest
        self.classifier.model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.classifier.train(X_train, y_train)
        
        # Evaluate with custom threshold (0.2)
        y_prob = self.classifier.predict_proba(X_test)  # Already 1D dropout probabilities
        y_pred_adjusted = [1 if prob > 0.2 else 0 for prob in y_prob]
        accuracy = accuracy_score(y_test, y_pred_adjusted)
        report = classification_report(y_test, y_pred_adjusted, target_names=["No Dropout", "Dropout"])
        print(f"Dropout Classifier Accuracy (threshold=0.2): {accuracy:.4f}")
        print("Classification Report (threshold=0.2):")
        print(report)
        
        # Update student profiles with probabilities
        probabilities = self.classifier.predict_proba(X)
        print("Sample Predicted Dropout Scores:")
        for student, prob in list(zip(students, probabilities))[:5]:
            print(f"- {student.student_id}: {prob:.4f} (Actual: {student.dropout_likelihood})")
        s00027_idx = next(i for i, s in enumerate(students) if s.student_id == "S00027")
        print(f"- S00027: {probabilities[s00027_idx]:.4f} (Actual: True)")
        
        for student, prob in zip(students, probabilities):
            student.predicted_dropout_score = prob

    def get_similar_students(self, student_id: str, limit: int = 5) -> List[StudentProfile]:
        target = self._get_student(student_id)
        if not target:
            return []
        similarities = []
        for student in self.students.values():
            if student.student_id != student_id:
                score = self.similarity_calculator.calculate_profile_similarity(target, student)
                similarities.append((student, score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:limit]
        print(f"Similar students to {student_id}:")
        for student, score in top_similar:
            print(f"- {student.student_id}: Similarity Score = {score:.4f}")
        return [student for student, _ in top_similar]

    def generate_recommendations(self, student_id: str, num_recommendations: int = 3) -> List[Recommendation]:
        student = self._get_student(student_id)
        if not student:
            return []
        similar_students = self.get_similar_students(student_id)
        available_courses = [
            course for course in self.courses.values()
            if course.course_name not in student.course_history
        ]
        return self.recommender.generate_recommendations(
            student=student,
            similar_students=similar_students,
            available_courses=available_courses,
            num_recommendations=num_recommendations
        )

    def _get_student(self, student_id: str) -> Optional[StudentProfile]:
        return self.students.get(student_id)

    def update_course_weights(self, course_name: str, weights: Dict[LearningStyle, float]) -> None:
        if course_name in self.courses:
            self.courses[course_name].content_type_weights = weights