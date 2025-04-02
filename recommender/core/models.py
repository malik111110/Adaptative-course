from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class EducationLevel(Enum):
    HIGH_SCHOOL = "High School"
    UNDERGRADUATE = "Undergraduate"
    POSTGRADUATE = "Postgraduate"

class EngagementLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class LearningStyle(Enum):
    VISUAL = "Visual"
    AUDITORY = "Auditory"
    READING_WRITING = "Reading/Writing"
    KINESTHETIC = "Kinesthetic"

@dataclass
class StudentProfile:
    student_id: str
    age: int
    gender: Gender
    education_level: EducationLevel
    learning_style: LearningStyle
    course_history: List[str]
    engagement_metrics: Dict[str, Dict[str, float]]
    quiz_attempts: Dict[str, int]
    engagement_level: EngagementLevel
    final_exam_scores: Dict[str, float]
    feedback_scores: Dict[str, int]
    dropout_likelihood: bool
    last_updated: datetime
    predicted_dropout_score: Optional[float] = None  # New: AI-predicted dropout probability

    def __post_init__(self):
        if isinstance(self.gender, str):
            self.gender = Gender(self.gender)
        if isinstance(self.education_level, str):
            self.education_level = EducationLevel(self.education_level)
        if isinstance(self.engagement_level, str):
            self.engagement_level = EngagementLevel(self.engagement_level)
        if isinstance(self.learning_style, str):
            self.learning_style = LearningStyle(self.learning_style)

@dataclass
class Course:
    course_name: str
    content_type_weights: Dict[LearningStyle, float]
    average_completion_rate: float
    average_quiz_score: float
    average_time_spent: float
    difficulty: Optional[float] = None

@dataclass
class Recommendation:
    course_name: str
    relevance_score: float
    reasoning: str