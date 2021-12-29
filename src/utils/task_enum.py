from enum import Enum

class Task(Enum):
    GRAPH_CLASSIFICATION = 1
    NODE_CLASSIFICATION = 2
    LINK_PREDICTION = 3

class Stage(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3
