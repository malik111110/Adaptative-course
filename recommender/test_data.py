from data.manager import DataManager

def test_data_loading():
    dataset_path = "/Users/mac/Desktop/Adaptative-courses/adaptative-course/personalized_learning_dataset.csv"
    data_manager = DataManager(dataset_path)
    
    try:
        print("Loading dataset...")
        data_manager.initialize()
        print("Dataset loaded successfully!")
        
        # Check a sample student
        sample_student = data_manager.get_student_profile("S00001")
        if sample_student:
            print(f"Sample student S00001: {sample_student.student_id}, "
                  f"Age: {sample_student.age}, "
                  f"Learning Style: {sample_student.learning_style.value}, "
                  f"Courses: {sample_student.course_history}")
        else:
            print("Student S00001 not found in dataset.")
        
        # Check total students and courses
        students = data_manager.get_all_students()
        courses = data_manager.get_all_courses()
        print(f"Total students loaded: {len(students)}")
        print(f"Total courses loaded: {len(courses)}")
        
        # Test a recommendation
        recommendations = data_manager.get_recommendations("S00001", num_recommendations=3)
        print("Recommendations for S00001:")
        for rec in recommendations:
            print(f"- {rec.course_name}: {rec.relevance_score:.2f}, {rec.reasoning}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_data_loading()