# This file contains configuration settings specific to the recommender system, such as parameters for algorithms and data sources.

class Config:
    DEBUG = True
    DATABASE_URI = 'sqlite:///recommender.db'
    RECOMMENDATION_ALGORITHM = 'collaborative_filtering'
    DATA_SOURCE = 'path/to/data/source'
    MAX_RECOMMENDATIONS = 10
    CACHE_TIMEOUT = 300  # seconds

class ProductionConfig(Config):
    DEBUG = False
    DATABASE_URI = 'postgresql://user:password@localhost/recommender'

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    DEBUG = False
    DATABASE_URI = 'sqlite:///:memory:'