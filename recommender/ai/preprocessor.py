from typing import List, Dict
from ..core.models import StudentProfile, Gender, EducationLevel, EngagementLevel, LearningStyle

class DataPreprocessor:
    def __init__(self):
        self.feature_columns = [
            "age", "gender", "education_level", "learning_style", "engagement_level",
            "avg_time_spent", "avg_quiz_attempts", "avg_quiz_score", "avg_forum_participation",
            "avg_assignment_completion", "avg_final_exam_score", "avg_feedback_score"
        ]

    def preprocess_students(self, students: List[StudentProfile]) -> List[List[float]]:
        """Convert student profiles to numerical feature vectors."""
        X = []
        for student in students:
            vector = self._vectorize_student(student)
            X.append(vector)
        return X

    def _vectorize_student(self, student: StudentProfile) -> List[float]:
        """Convert a single student profile to a numerical vector."""
        vector = []
        
        # Age (normalized)
        vector.append(student.age / 50.0)
        
        # Gender (one-hot)
        vector.extend([
            1.0 if student.gender == Gender.MALE else 0.0,
            1.0 if student.gender == Gender.FEMALE else 0.0,
            1.0 if student.gender == Gender.OTHER else 0.0
        ])
        
        # Education Level (one-hot)
        vector.extend([
            1.0 if student.education_level == EducationLevel.HIGH_SCHOOL else 0.0,
            1.0 if student.education_level == EducationLevel.UNDERGRADUATE else 0.0,
            1.0 if student.education_level == EducationLevel.POSTGRADUATE else 0.0
        ])
        
        # Learning Style (one-hot)
        vector.extend([
            1.0 if student.learning_style == LearningStyle.VISUAL else 0.0,
            1.0 if student.learning_style == LearningStyle.AUDITORY else 0.0,
            1.0 if student.learning_style == LearningStyle.READING_WRITING else 0.0,
            1.0 if student.learning_style == LearningStyle.KINESTHETIC else 0.0
        ])
        
        # Engagement Level (ordinal)
        vector.append(
            1.0 if student.engagement_level == EngagementLevel.HIGH else
            0.5 if student.engagement_level == EngagementLevel.MEDIUM else 0.0
        )
        
        # Average engagement metrics
        avg_metrics = self._average_engagement_metrics(student.engagement_metrics)
        vector.append(avg_metrics.get("time_spent_on_videos", 0.0) / 500.0)
        vector.append(sum(student.quiz_attempts.values()) / max(1, len(student.quiz_attempts)) / 4.0)
        vector.append(avg_metrics.get("quiz_scores", 0.0) / 100.0)
        vector.append(avg_metrics.get("forum_participation", 0.0) / 50.0)
        vector.append(avg_metrics.get("assignment_completion_rate", 0.0) / 100.0)
        
        # Average performance metrics
        vector.append(sum(student.final_exam_scores.values()) / max(1, len(student.final_exam_scores)) / 100.0)
        vector.append(sum(student.feedback_scores.values()) / max(1, len(student.feedback_scores)) / 5.0)
        
        return vector

    def _average_engagement_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {"time_spent_on_videos": 0.0, "quiz_scores": 0.0, "forum_participation": 0.0, "assignment_completion_rate": 0.0}
        avg_metrics = {
            "time_spent_on_videos": 0.0,
            "quiz_scores": 0.0,
            "forum_participation": 0.0,
            "assignment_completion_rate": 0.0
        }
        n = len(metrics)
        for course_metrics in metrics.values():
            for key in avg_metrics:
                avg_metrics[key] += course_metrics.get(key, 0.0)
        for key in avg_metrics:
            avg_metrics[key] /= n
        return avg_metrics