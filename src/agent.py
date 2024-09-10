import sys, os

sys.path.append(sys.path[0])
from src.knowledge_graph import NetWeaver
from src.openai_inference import Inference
from src.utilities import *

class MyAgent:
    def __init__(self):
        self.OAI = Inference(self)   ### OPENAI
        self.KG = NetWeaver(self)   ### KNOWLEDGE-GRAPH