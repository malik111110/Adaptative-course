from typing import List, Dict
from ..core.models import StudentProfile, Course, Recommendation, LearningStyle

class CourseRecommender:
    def __init__(self):
        pass

    def generate_recommendations(
        self,
        student: StudentProfile,
        similar_students: List[StudentProfile],
        available_courses: List[Course],
        num_recommendations: int = 3
    ) -> List[Recommendation]:
        """Generate course recommendations based on student profile and similar students."""
        recommendations = []
        
        for course in available_courses:
            # Content match score based on learning style
            content_match = course.content_type_weights.get(student.learning_style, 0.0)
            
            # Collaborative filtering: success rate among similar students
            similar_students_success = [
                s.final_exam_scores.get(course.course_name, 0.0) / 100.0
                for s in similar_students
                if course.course_name in s.course_history and s.final_exam_scores.get(course.course_name, 0.0) > 0
            ]
            collab_score = sum(similar_students_success) / len(similar_students_success) if similar_students_success else 0.0
            collab_count = len(similar_students_success)
            
            # Base relevance score
            relevance_score = 0.5 * content_match + 0.5 * collab_score
            
            # Adjust for high dropout risk
            dropout_adjustment = 0.0
            if student.predicted_dropout_score and student.predicted_dropout_score > 0.5:
                dropout_adjustment = 0.25  # Fixed adjustment for high risk
                relevance_score += dropout_adjustment
            
            # Reasoning
            reasoning_parts = [f"Matches learning style ({student.learning_style.value}: {content_match:.2f})"]
            if collab_count > 0:
                reasoning_parts.append(f"Popular among {collab_count} similar students (avg success: {collab_score:.2f})")
            if dropout_adjustment > 0:
                reasoning_parts.append(f"Adjusted for high dropout risk (+{dropout_adjustment:.2f})")
            reasoning = ". ".join(reasoning_parts) + "."
            
            recommendations.append(Recommendation(
                course_name=course.course_name,
                reasoning=reasoning,
                relevance_score=relevance_score
            ))
        
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations[:num_recommendations]