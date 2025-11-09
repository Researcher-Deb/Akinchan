"""Google ADK agents using Gemini 2.0 Flash."""
import logging
from datetime import datetime

try:
    import google.generativeai as genai
    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False

from .models import AgentRequest, AgentResponse
from .config import get_settings
from .database import get_db

logger = logging.getLogger(__name__)
settings = get_settings()

class GoogleADKAgent:
    def __init__(self, agent_type, instruction):
        self.agent_type = agent_type
        self.db = get_db()
        self.model = None
        if GOOGLE_ADK_AVAILABLE and settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME, system_instruction=instruction)
    
    def process(self, request):
        if not self.model:
            raise RuntimeError(f"Google ADK agent not initialized. ADK available: {GOOGLE_ADK_AVAILABLE}, API key set: {bool(settings.GOOGLE_API_KEY)}")
        
        prompt = request.query
        if request.context:
            stats = self.db.get_historical_stats()
            prompt += f"\n\nBenchmark: {stats.get('success_rate',0)*100:.1f}% success rate"
        
        response = self.model.generate_content(prompt)
        return AgentResponse(agent_type=self.agent_type, response=response.text, confidence=0.92, sources=["Google ADK"], timestamp=datetime.now())

class ResearchAgent(GoogleADKAgent):
    def __init__(self):
        super().__init__("research", "Clinical trial research analyst")

class PredictionAgent(GoogleADKAgent):
    def __init__(self):
        super().__init__("prediction", "Trial prediction specialist")

class OptimizationAgent(GoogleADKAgent):
    def __init__(self):
        super().__init__("optimization", "Trial optimization expert")

class ReportAgent(GoogleADKAgent):
    def __init__(self):
        super().__init__("report", "Trial reporting specialist")

class AgentFactory:
    _agents = None
    @classmethod
    def _init(cls):
        if cls._agents is None:
            cls._agents = {'research': ResearchAgent(), 'prediction': PredictionAgent(), 'optimization': OptimizationAgent(), 'report': ReportAgent()}
    @classmethod
    def get_agent(cls, agent_type):
        cls._init()
        return cls._agents.get(agent_type)
    @classmethod
    def process_request(cls, request):
        agent = cls.get_agent(request.agent_type)
        if not agent:
            raise ValueError(f"Unknown agent: {request.agent_type}")
        return agent.process(request)
