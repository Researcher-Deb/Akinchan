"""
Agent Router: Intelligent routing between Google ADK and Fallback agents.

This router provides automatic failover between:
1. Primary: Google ADK agents (agents.py) - Uses Gemini 2.0 Flash
2. Fallback: Rule-based agents (fallback_agents.py) - Always available

Routing Logic:
- If Google ADK available + API key configured → Use agents.py
- If ADK fails at runtime → Automatically fallback to fallback_agents.py
- If ADK not available → Use fallback_agents.py directly
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .models import AgentRequest, AgentResponse
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Try to import Google ADK agents
try:
    from . import agents as adk_agents
    ADK_AGENTS_AVAILABLE = hasattr(adk_agents, 'GOOGLE_ADK_AVAILABLE') and adk_agents.GOOGLE_ADK_AVAILABLE
    logger.info(f"✓ Google ADK agents module loaded (ADK available: {ADK_AGENTS_AVAILABLE})")
except Exception as e:
    ADK_AGENTS_AVAILABLE = False
    logger.warning(f"⚠ Could not load Google ADK agents: {e}")
    adk_agents = None

# Import fallback agents
try:
    from . import fallback_agents
    FALLBACK_AGENTS_AVAILABLE = True
    logger.info("✓ Fallback agents module loaded")
except Exception as e:
    FALLBACK_AGENTS_AVAILABLE = False
    fallback_agents = None
    logger.error(f"❌ Could not load fallback agents: {e}")


class AgentRouter:
    """
    Intelligent router for agent requests.
    
    Automatically routes to:
    1. Google ADK agents (Gemini 2.0 Flash) when available
    2. Fallback agents (rule-based) as backup
    
    Provides seamless failover and detailed logging.
    """
    
    _use_adk = None
    _routing_decision_made = False
    
    @classmethod
    def _determine_routing_strategy(cls):
        """Determine which agent system to use."""
        if cls._routing_decision_made:
            return cls._use_adk
        
        # Check if Google ADK is available and configured
        if ADK_AGENTS_AVAILABLE and settings.GOOGLE_API_KEY:
            cls._use_adk = True
            logger.info("🤖 Agent Router: Using Google ADK (Gemini 2.0 Flash)")
        else:
            cls._use_adk = False
            if not ADK_AGENTS_AVAILABLE:
                logger.info("📋 Agent Router: Google ADK not available, using fallback")
            elif not settings.GOOGLE_API_KEY:
                logger.info("📋 Agent Router: GOOGLE_API_KEY not set, using fallback")
        
        cls._routing_decision_made = True
        return cls._use_adk
    
    @classmethod
    def process_request(cls, request: AgentRequest) -> AgentResponse:
        """
        Process agent request with automatic routing and fallback.
        
        Flow:
        1. Determine routing strategy (ADK vs Fallback)
        2. Try primary route
        3. On error, fallback to secondary route
        4. If both fail, return error response
        
        Args:
            request: AgentRequest with query, agent_type, and context
            
        Returns:
            AgentResponse with results from either ADK or fallback agents
        """
        use_adk = cls._determine_routing_strategy()
        
        # Try Google ADK first if available
        if use_adk and adk_agents:
            try:
                logger.debug(f"→ Routing to Google ADK: {request.agent_type}")
                response = adk_agents.AgentFactory.process_request(request)
                logger.debug(f"✓ ADK response received ({len(response.response)} chars)")
                return response
            except Exception as e:
                logger.error(f"❌ Google ADK processing failed: {e}")
                logger.info("↩ Falling back to rule-based agents")
                # Fall through to fallback
        
        # Use fallback agents
        if FALLBACK_AGENTS_AVAILABLE and fallback_agents:
            try:
                logger.debug(f"→ Routing to fallback agents: {request.agent_type}")
                response = fallback_agents.AgentFactory.process_request(request)
                logger.debug(f"✓ Fallback response received")
                return response
            except Exception as e:
                logger.error(f"❌ Fallback agent processing failed: {e}")
        
        # Both systems failed - return error
        return AgentResponse(
            agent_type=request.agent_type,
            response=f"⚠ Agent processing unavailable. Both Google ADK and fallback systems failed. Please check logs and configuration.",
            confidence=0.0,
            sources=["Error Handler"],
            timestamp=datetime.now()
        )
    
    @classmethod
    def get_agent(cls, agent_type: str):
        """
        Get agent instance (for compatibility with existing code).
        
        Args:
            agent_type: Type of agent ('research', 'prediction', 'optimization', 'report')
            
        Returns:
            Agent instance from appropriate system
        """
        use_adk = cls._determine_routing_strategy()
        
        if use_adk and adk_agents:
            try:
                return adk_agents.AgentFactory.get_agent(agent_type)
            except Exception as e:
                logger.warning(f"Failed to get ADK agent: {e}")
        
        if FALLBACK_AGENTS_AVAILABLE and fallback_agents:
            return fallback_agents.AgentFactory.get_agent(agent_type)
        
        raise RuntimeError(f"No agent system available for type: {agent_type}")
    
    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """
        Get comprehensive status of agent routing system.
        
        Returns:
            Dictionary with routing status, availability, and configuration
        """
        use_adk = cls._determine_routing_strategy()
        
        status = {
            "router_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            # Flat keys for easy access
            "google_adk_available": ADK_AGENTS_AVAILABLE,
            "api_key_configured": bool(settings.GOOGLE_API_KEY),
            "active_system": "Google ADK" if use_adk else "Fallback",
            "fallback_available": FALLBACK_AGENTS_AVAILABLE,
            # Nested structure for detailed info
            "routing": {
                "active_system": "Google ADK" if use_adk else "Fallback",
                "primary_available": ADK_AGENTS_AVAILABLE,
                "fallback_available": FALLBACK_AGENTS_AVAILABLE
            },
            "google_adk": {
                "module_loaded": adk_agents is not None,
                "adk_available": ADK_AGENTS_AVAILABLE,
                "api_key_configured": bool(settings.GOOGLE_API_KEY),
                "model": settings.GEMINI_MODEL_NAME if ADK_AGENTS_AVAILABLE else None
            },
            "fallback": {
                "module_loaded": fallback_agents is not None,
                "available": FALLBACK_AGENTS_AVAILABLE
            },
            "agents": {
                "research": "available",
                "prediction": "available",
                "optimization": "available",
                "report": "available"
            }
        }
        
        # Add per-agent routing info
        for agent_type in ["research", "prediction", "optimization", "report"]:
            try:
                agent = cls.get_agent(agent_type)
                status["agents"][agent_type] = {
                    "available": True,
                    "using_adk": use_adk,
                    "source": "Google ADK" if use_adk else "Fallback"
                }
            except Exception as e:
                status["agents"][agent_type] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
    @classmethod
    def force_fallback_mode(cls):
        """Force router to use fallback mode (useful for testing)."""
        cls._use_adk = False
        cls._routing_decision_made = True
        logger.warning("⚠ Forced fallback mode activated")
    
    @classmethod
    def reset_routing_decision(cls):
        """Reset routing decision (re-evaluate on next request)."""
        cls._routing_decision_made = False
        logger.info("↻ Routing decision reset")


# Factory alias for backward compatibility
class AgentFactory:
    """Alias for AgentRouter to maintain backward compatibility."""
    
    @classmethod
    def process_request(cls, request: AgentRequest) -> AgentResponse:
        return AgentRouter.process_request(request)
    
    @classmethod
    def get_agent(cls, agent_type: str):
        return AgentRouter.get_agent(agent_type)
    
    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        return AgentRouter.get_status()


# Export main classes
__all__ = [
    'AgentRouter',
    'AgentFactory',  # Backward compatibility
    'ADK_AGENTS_AVAILABLE',
    'FALLBACK_AGENTS_AVAILABLE'
]
