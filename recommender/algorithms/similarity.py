from typing import Dict, List
import math
from ..core.models import StudentProfile, LearningStyle, Gender, EducationLevel, EngagementLevel

class CosineSimilarity:
    def calculate_profile_similarity(self, student1: StudentProfile, student2: StudentProfile) -> float:
        """Calculate cosine similarity between two student profiles."""
        vector1 = self._vectorize_profile(student1)
        vector2 = self._vectorize_profile(student2)
        
        if not vector1 or not vector2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def _vectorize_profile(self, profile: StudentProfile) -> List[float]:
        """Convert a student profile into a numerical vector for similarity calculation."""
        vector = []

        # 1. Learning Style (one-hot encoding)
        learning_style_vector = [
            1.0 if profile.learning_style == LearningStyle.VISUAL else 0.0,
            1.0 if profile.learning_style == LearningStyle.AUDITORY else 0.0,
            1.0 if profile.learning_style == LearningStyle.READING_WRITING else 0.0,
            1.0 if profile.learning_style == LearningStyle.KINESTHETIC else 0.0
        ]
        vector.extend(learning_style_vector)

        # 2. Demographics
        vector.append(profile.age / 50.0)  # Normalize age (assuming max 50)
        vector.append(1.0 if profile.gender == Gender.MALE else 0.5 if profile.gender == Gender.FEMALE else 0.0)
        vector.append(1.0 if profile.education_level == EducationLevel.POSTGRADUATE else 
                     0.5 if profile.education_level == EducationLevel.UNDERGRADUATE else 0.0)

        # 3. Engagement Level (ordinal encoding)
        engagement_value = (1.0 if profile.engagement_level == EngagementLevel.HIGH else 
                           0.5 if profile.engagement_level == EngagementLevel.MEDIUM else 0.0)
        vector.append(engagement_value)

        # 4. Engagement Metrics (averaged across courses)
        avg_metrics = self._average_engagement_metrics(profile.engagement_metrics)
        vector.append(avg_metrics.get("time_spent_on_videos", 0.0) / 500.0)  # Normalize (max ~500)
        vector.append(avg_metrics.get("quiz_scores", 0.0) / 100.0)           # Normalize (0-100)
        vector.append(avg_metrics.get("forum_participation", 0.0) / 50.0)   # Normalize (max ~50)
        vector.append(avg_metrics.get("assignment_completion_rate", 0.0) / 100.0)  # Normalize (0-100)

        # 5. Performance Metrics (averaged across courses)
        avg_quiz_attempts = sum(profile.quiz_attempts.values()) / max(1, len(profile.quiz_attempts)) / 4.0  # Normalize (max 4)
        avg_final_exam = sum(profile.final_exam_scores.values()) / max(1, len(profile.final_exam_scores)) / 100.0  # Normalize (0-100)
        avg_feedback = sum(profile.feedback_scores.values()) / max(1, len(profile.feedback_scores)) / 5.0  # Normalize (1-5)
        vector.extend([avg_quiz_attempts, avg_final_exam, avg_feedback])

        # 6. Dropout Likelihood
        vector.append(1.0 if profile.dropout_likelihood else 0.0)

        return vector

    def _average_engagement_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average engagement metrics across all courses."""
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