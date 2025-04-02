import csv
from typing import List, Dict
from pathlib import Path
from ..core.services import RecommendationService

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.service = RecommendationService()

    def load_dataset(self) -> None:
        """Load the CSV dataset and populate the RecommendationService."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")

        with open(self.file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.service.load_student_from_csv_row(row)

    def get_service(self) -> RecommendationService:
        """Return the populated RecommendationService."""
        return self.service