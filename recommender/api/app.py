from flask import Flask, request
from flask_restx import Api, Resource, fields
from recommender.data.manager import DataManager
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Define dataset path
DATASET_PATH = "/Users/mac/Desktop/Adaptative-courses/personalized_learning_dataset.csv"
data_manager = DataManager(DATASET_PATH)
data_manager.initialize()

# Initialize Flask-RESTX API with Swagger configuration
api = Api(
    app,
    version='1.0',
    title='Adaptive Courses API',
    description='A RESTful API for personalized course recommendations, student profiles, and analytics in an online learning platform.',
    doc='/swagger-ui'  # Swagger UI endpoint
)

# Namespace for organizing endpoints
ns = api.namespace('api', description='Main API operations')

# Define response models for Swagger documentation
student_model = api.model('Student', {
    'student_id': fields.String(example='S00027', description='Unique student identifier'),
    'age': fields.Integer(example=30, description='Student age (18-40)'),
    'gender': fields.String(example='Male', enum=['Male', 'Female', 'Other'], description='Student gender'),
    'education_level': fields.String(example='High School', enum=['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], description='Education level'),
    'learning_style': fields.String(example='Visual', enum=['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic'], description='Preferred learning style'),
    'course_history': fields.List(fields.String, example=['Python Basics'], description='List of courses taken by the student'),
    'engagement_level': fields.String(example='Medium', enum=['Low', 'Medium', 'High'], description='Engagement level'),
    'dropout_likelihood': fields.Boolean(example=True, description='Actual dropout status (Yes/No)'),
    'predicted_dropout_score': fields.Float(example=0.7, description='Predicted dropout probability (0-1, rounded to 4 decimals)')
})

recommendation_model = api.model('Recommendation', {
    'course_name': fields.String(example='Web Development', description='Name of the recommended course'),
    'relevance_score': fields.Float(example=0.87, description='Relevance score (0-1+, rounded to 2 decimals)'),
    'reasoning': fields.String(example='Matches learning style (Visual: 0.40). Popular among 3 similar students (avg success: 0.84). Adjusted for high dropout risk (+0.25).', description='Explanation of why the course is recommended')
})

recommendations_response = api.model('RecommendationsResponse', {
    'student_id': fields.String(example='S00027', description='Unique student identifier'),
    'recommendations': fields.List(fields.Nested(recommendation_model), description='List of recommended courses')
})

course_model = api.model('Course', {
    'course_name': fields.String(example='Python Basics', description='Name of the course'),
    'average_completion_rate': fields.Float(example=85.5, description='Average assignment completion rate (%)'),
    'average_quiz_score': fields.Float(example=75.2, description='Average quiz score (0-100)'),
    'average_time_spent': fields.Float(example=10.3, description='Average time spent on videos (hours)')
})

courses_response = api.model('CoursesResponse', {
    'courses': fields.List(fields.Nested(course_model), description='List of all available courses')
})

health_model = api.model('Health', {
    'status': fields.String(example='healthy', description='API health status'),
    'message': fields.String(example='API is running', description='Health check message')
})

analysis_model = api.model('Analysis', {
    'avg_dropout_risk': fields.Float(example=0.1976, description='Average predicted dropout risk across all students (0-1)'),
    'course_statistics': fields.Raw(example={'Python Basics': {'avg_quiz_score': 75.2, 'avg_completion_rate': 85.5}}, description='Statistics per course (quiz scores, completion rates, etc.)'),
    'dropout_risk_distribution': fields.Raw(example={'0-0.25': 9995, '0.25-0.5': 5, '0.5-0.75': 0, '0.75-1': 0}, description='Distribution of dropout risk scores'),
    'engagement_distribution': fields.Raw(example={'High': 2980, 'Medium': 4927, 'Low': 2093}, description='Distribution of engagement levels'),
    'total_students': fields.Integer(example=10000, description='Total number of students')
})

error_model = api.model('Error', {
    'error': fields.String(example='Student S00027 not found', description='Error message')
})

# Endpoint definitions with Swagger documentation
@ns.route('/students/<string:student_id>')
class StudentResource(Resource):
    @ns.doc(description='Retrieve a student profile by ID.')
    @ns.response(200, 'Success', student_model)
    @ns.response(404, 'Student not found', error_model)
    def get(self, student_id):
        """Get student profile details"""
        student = data_manager.get_student_profile(student_id)
        if not student:
            return {'error': f"Student {student_id} not found"}, 404
        student_data = {
            "student_id": student.student_id,
            "age": student.age,
            "gender": student.gender.value,
            "education_level": student.education_level.value,
            "learning_style": student.learning_style.value,
            "course_history": student.course_history,
            "engagement_level": student.engagement_level.value,
            "dropout_likelihood": student.dropout_likelihood,
            "predicted_dropout_score": round(student.predicted_dropout_score, 4) if student.predicted_dropout_score is not None else None
        }
        return student_data, 200

@ns.route('/recommendations/<string:student_id>')
class RecommendationsResource(Resource):
    @ns.doc(description='Get course recommendations for a student by ID.')
    @ns.param('num', 'Number of recommendations (default: 3)', type=int, default=3, required=False)
    @ns.response(200, 'Success', recommendations_response)
    @ns.response(404, 'Student not found', error_model)
    def get(self, student_id):
        """Get personalized course recommendations"""
        num_recommendations = request.args.get('num', default=3, type=int)
        recommendations = data_manager.get_recommendations(student_id, num_recommendations)
        if not recommendations and not data_manager.get_student_profile(student_id):
            return {'error': f"Student {student_id} not found"}, 404
        recommendations_data = [
            {
                "course_name": rec.course_name,
                "relevance_score": round(rec.relevance_score, 2),
                "reasoning": rec.reasoning
            }
            for rec in recommendations
        ]
        return {"student_id": student_id, "recommendations": recommendations_data}, 200

@ns.route('/courses')
class CoursesResource(Resource):
    @ns.doc(description='Retrieve a list of all available courses.')
    @ns.response(200, 'Success', courses_response)
    def get(self):
        """Get all courses with their statistics"""
        courses = data_manager.get_all_courses()
        return {"courses": courses}, 200

@ns.route('/health')
class HealthResource(Resource):
    @ns.doc(description='Check the health status of the API.')
    @ns.response(200, 'API is healthy', health_model)
    def get(self):
        """Health check endpoint"""
        return {"status": "healthy", "message": "API is running"}, 200

@ns.route('/analysis')
class AnalysisResource(Resource):
    @ns.doc(description='Get aggregated analysis data for dropout risk, engagement, and course performance.')
    @ns.response(200, 'Success', analysis_model)
    def get(self):
        """Get analysis data for dashboard insights"""
        analysis_data = data_manager.get_analysis_data()
        return analysis_data, 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)