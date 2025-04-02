import unittest
from recommender.algorithms.collaborative_filtering import CollaborativeFiltering
from recommender.algorithms.content_based import ContentBasedFiltering

class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        self.collab_filter = CollaborativeFiltering()
        self.content_filter = ContentBasedFiltering()

    def test_collaborative_filtering(self):
        # Example test case for collaborative filtering
        user_id = 1
        recommendations = self.collab_filter.get_recommendations(user_id)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_content_based_filtering(self):
        # Example test case for content-based filtering
        item_id = 1
        recommendations = self.content_filter.get_recommendations(item_id)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()