from typing import Optional, List, Dict
from ..core.models import StudentProfile, Recommendation
from ..core.services import RecommendationService
from .loader import DataLoader

class DataManager:
    def __init__(self, dataset_path: str):
        self.loader = DataLoader(dataset_path)
        self.service: Optional[RecommendationService] = None

    def initialize(self) -> None:
        self.loader.load_dataset()
        self.service = self.loader.get_service()
        self.service.train_classifier()

    def get_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        if not self.service:
            raise ValueError("DataManager not initialized. Call initialize() first.")
        return self.service._get_student(student_id)

    def get_recommendations(self, student_id: str, num_recommendations: int = 3) -> List[Recommendation]:
        if not self.service:
            raise ValueError("DataManager not initialized. Call initialize() first.")
        return self.service.generate_recommendations(student_id, num_recommendations)

    def get_all_students(self) -> List[StudentProfile]:
        if not self.service:
            raise ValueError("DataManager not initialized. Call initialize() first.")
        return list(self.service.students.values())

    def get_all_courses(self) -> List[str]:
        if not self.service:
            raise ValueError("DataManager not initialized. Call initialize() first.")
        return list(self.service.courses.keys())

    def get_analysis_data(self) -> Dict:
        """Generate detailed analysis data for dashboard service."""
        if not self.service:
            raise ValueError("DataManager not initialized. Call initialize() first.")
        students = self.get_all_students()
        dropout_risk = [s.predicted_dropout_score for s in students if s.predicted_dropout_score is not None]
        engagement_levels = [s.engagement_level.value for s in students]
        
        # Dropout risk distribution
        dropout_bins = {"0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}
        for risk in dropout_risk:
            if risk <= 0.25:
                dropout_bins["0-0.25"] += 1
            elif risk <= 0.5:
                dropout_bins["0.25-0.5"] += 1
            elif risk <= 0.75:
                dropout_bins["0.5-0.75"] += 1
            else:
                dropout_bins["0.75-1.0"] += 1
        
        # Per-course statistics
        course_stats = {}
        for course_name, course in self.service.courses.items():
            course_students = [s for s in students if course_name in s.course_history]
            if course_students:
                course_dropout_risk = [s.predicted_dropout_score for s in course_students if s.predicted_dropout_score is not None]
                course_stats[course_name] = {
                    "student_count": len(course_students),
                    "avg_dropout_risk": sum(course_dropout_risk) / len(course_dropout_risk) if course_dropout_risk else 0.0,
                    "avg_final_exam_score": sum(s.final_exam_scores.get(course_name, 0.0) for s in course_students) / len(course_students)
                }

        return {
            "total_students": len(students),
            "avg_dropout_risk": sum(dropout_risk) / len(dropout_risk) if dropout_risk else 0.0,
            "engagement_distribution": {
                "Low": engagement_levels.count("Low"),
                "Medium": engagement_levels.count("Medium"),
                "High": engagement_levels.count("High")
            },
            "dropout_risk_distribution": dropout_bins,
            "course_statistics": course_stats
        }