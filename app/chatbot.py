"""
Multilingual Chat Service for Clinical Trial Simulator
Supports conversation, simulation updates, and data translation
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import google.generativeai as genai
from .config import get_settings
from .database import CSVDatabase

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """
    Multilingual chatbot service powered by Gemini 2.5 Flash.
    Supports English, Hindi, Bengali, and other languages.
    """
    
    def __init__(self, user_id: str, username: str, database: CSVDatabase):
        self.user_id = user_id
        self.username = username
        self.db = database
        self.context = []
        self.language = None
        self.pending_action = None
        self.conversation_history = []
        self.current_simulation_id = None  # Track current simulation being discussed
        
        # Initialize Gemini with fallback support
        self.model = None
        self.api_key_used = None
        self.primary_key = settings.GEMINI_API_KEY
        self.fallback_key = getattr(settings, 'GOOGLE_API_KEY', None)
        
        # Try to initialize with primary key
        self._initialize_model()
        
        # If both failed, raise error
        if self.model is None:
            error_msg = "No working Gemini API key available. Please check GEMINI_API_KEY or GOOGLE_API_KEY in .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_model(self, use_fallback: bool = False):
        """Initialize Gemini model with primary or fallback key."""
        
        # Determine which key to use
        if use_fallback:
            api_key = self.fallback_key
            key_name = "GOOGLE_API_KEY (fallback)"
        else:
            api_key = self.primary_key
            key_name = "GEMINI_API_KEY (primary)"
        
        # Skip if key is not set or is placeholder
        if not api_key or api_key.startswith("YOUR_NEW_API_KEY"):
            if not use_fallback:
                logger.warning(f"⚠️ {key_name} not configured, trying fallback...")
                self._initialize_model(use_fallback=True)
            return
        
        # Try to configure and create model
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            self.api_key_used = "fallback" if use_fallback else "primary"
            logger.info(f"✅ Chat service initialized with {key_name} for user: {self.username}")
        except Exception as e:
            logger.warning(f"⚠️ {key_name} configuration failed: {str(e)[:100]}")
            self.model = None
            
            # If primary failed, try fallback
            if not use_fallback and self.fallback_key and self.fallback_key != self.primary_key:
                logger.info("🔄 Attempting to use fallback API key...")
                self._initialize_model(use_fallback=True)
    
    def _call_gemini_with_fallback(self, prompt: str) -> str:
        """
        Call Gemini API with automatic fallback to secondary key if primary fails.
        Returns the response text.
        """
        max_retries = 2  # Try primary, then fallback
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # If this is a retry and we were using primary, switch to fallback
                if attempt > 0 and self.api_key_used == "primary":
                    logger.warning("🔄 Primary key failed, switching to fallback key...")
                    self._initialize_model(use_fallback=True)
                    
                    if self.model is None:
                        raise ValueError("Fallback API key initialization failed")
                
                # Make the API call
                logger.info(f"Calling Gemini API (using {self.api_key_used} key, attempt {attempt + 1}/{max_retries})")
                response = self.model.generate_content(prompt)
                
                # Success!
                logger.info(f"✅ Gemini API response received: {len(response.text)} chars")
                return response.text
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.error(f"❌ Gemini API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}")
                
                # If this is the last attempt, give up
                if attempt >= max_retries - 1:
                    break
        
        # All attempts failed
        raise Exception(f"Gemini API failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    def _get_system_prompt(self, user_simulations: List[Dict]) -> str:
        """Generate system prompt with user's simulation context."""
        
        sim_list = "\n".join([
            f"- ID: {sim['simulation_id']}, "
            f"Patients Enrolled: {sim.get('patients_enrolled', 'N/A')}, "
            f"Patients Completed: {sim.get('patients_completed', 'N/A')}, "
            f"Success Probability: {sim.get('success_probability', 'N/A')}, "
            f"Cost: ${sim.get('estimated_cost', 'N/A')}"
            for sim in user_simulations
        ])
        
        # Add current simulation context if available
        current_context = ""
        if self.current_simulation_id:
            current_context = f"\n\n**CURRENT SIMULATION IN DISCUSSION:** {self.current_simulation_id}\n(User is currently asking about this simulation. Use it as default when they say 'this simulation', 'it', or don't specify which one.)"
        
        return f"""You are a multilingual AI assistant for a Clinical Trial Simulation platform.

**User Information:**
- Username: {self.username}
- User ID: {self.user_id}
- Language: {self.language or 'Auto-detect from user message'}

**User's Simulations:**
{sim_list if sim_list else "No simulations yet"}{current_context}

**Your Capabilities:**
1. Answer questions about clinical trials and simulations
2. Help users view their simulation data
3. Assist with updating INPUT parameters (patient count, drug dosage, trial duration, etc.)
4. Translate simulation descriptions and data to different languages
5. Explain simulation results and statistics

**Important Rules:**
1. ALWAYS respond in the SAME LANGUAGE the user uses
2. For data updates, show: Field name, Current value, New value
3. ALWAYS ask for confirmation before making any changes
4. Users can ONLY update INPUT fields, NOT output/results
5. After confirming update, trigger re-simulation automatically
6. Be helpful, friendly, and professional
7. If unsure, ask clarifying questions
8. Use simple, clean formatting - avoid excessive asterisks and markdown
9. Be concise and direct in responses

**Formatting Guidelines:**
- Use bullet points with simple dashes (-)
- Avoid bold (**) and italic (*) unless absolutely necessary
- Use emojis sparingly (only for status indicators like ✓, ❌, 📝)
- Keep responses clean and readable

**Update Format:**
When user wants to update, respond with:
```
📝 Update Request:
- Simulation ID: [ID]
- Field: [field_name]
- Current Value: [old_value]
- New Value: [new_value]

Reply with 'YES' or 'CONFIRM' to proceed, or 'NO' or 'CANCEL' to abort.
```

**Translation Format:**
When translating, show original and translated side-by-side.

Remember: Be conversational, helpful, and use clean formatting!"""
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process user message and generate response.
        
        Returns:
            {
                'response': str,
                'language': str,
                'action_type': str,  # 'chat', 'update_request', 'confirmation', 'translation'
                'action_data': dict or None
            }
        """
        try:
            logger.info(f"Processing message from {self.username}: {message[:50]}...")
            
            # Detect language if first message
            if not self.language:
                self.language = self._detect_language(message)
                logger.info(f"Detected language: {self.language}")
            
            # Extract and track simulation ID mentioned in message
            sim_id_in_message = self._extract_simulation_id(message)
            if sim_id_in_message:
                self.current_simulation_id = sim_id_in_message
                logger.info(f"Tracking simulation: {sim_id_in_message}")
            
            # Check if this is a confirmation response
            if self.pending_action and self._is_confirmation(message):
                return await self._handle_confirmation(message)
            
            # Check for update intent
            update_intent = self._parse_update_intent(message)
            if update_intent:
                return await self._handle_update_request(update_intent)
            
            # Check for translation request
            translation_intent = self._parse_translation_intent(message)
            if translation_intent:
                return await self._handle_translation_request(translation_intent)
            
            # Regular conversation
            return await self._handle_chat(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return {
                'response': f"Sorry, I encountered an error: {str(e)}",
                'language': self.language or 'en',
                'action_type': 'error',
                'action_data': None
            }
    
    def _extract_simulation_id(self, message: str) -> Optional[str]:
        """Extract simulation ID from message if mentioned."""
        # Get user's simulations
        user_sims = self.db.get_user_simulations(self.user_id)
        
        # Check if any simulation ID is mentioned in the message
        for sim in user_sims:
            sim_id = sim['simulation_id']
            if sim_id.lower() in message.lower():
                return sim_id
        
        return None
    
    def _detect_language(self, message: str) -> str:
        """Detect language from message using simple heuristics and Gemini."""
        
        # Simple detection based on character sets
        if any('\u0900' <= char <= '\u097F' for char in message):
            return 'hi'  # Hindi (Devanagari script)
        elif any('\u0980' <= char <= '\u09FF' for char in message):
            return 'bn'  # Bengali
        else:
            return 'en'  # Default to English
    
    def _is_confirmation(self, message: str) -> bool:
        """Check if message is a confirmation (yes/no)."""
        msg_lower = message.lower().strip()
        
        # English
        if msg_lower in ['yes', 'confirm', 'ok', 'proceed', 'y']:
            return True
        if msg_lower in ['no', 'cancel', 'abort', 'n']:
            return True
        
        # Hindi
        if msg_lower in ['हां', 'हाँ', 'जी', 'ठीक है', 'हां करें']:
            return True
        if msg_lower in ['नहीं', 'रद्द करें', 'नही']:
            return True
        
        # Bengali
        if msg_lower in ['হ্যাঁ', 'ঠিক আছে', 'করুন']:
            return True
        if msg_lower in ['না', 'বাতিল', 'নাহ']:
            return True
        
        return False
    
    def _parse_update_intent(self, message: str) -> Optional[Dict]:
        """Parse message for update intent using patterns and keywords."""
        
        msg_lower = message.lower()
        
        # Check for "set it to X" or "change to X" patterns first
        set_to_patterns = [
            r'set\s+(?:it\s+)?to\s+(\d+)',
            r'change\s+(?:it\s+)?to\s+(\d+)',
            r'make\s+(?:it\s+)?(\d+)',
            r'update\s+(?:it\s+)?to\s+(\d+)',
        ]
        
        for pattern in set_to_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                # User said "set it to 202" - infer patient number from context
                return {
                    'simulation_id': self.current_simulation_id,  # Use context
                    'field': 'patients_enrolled',  # Assume patient count
                    'new_value': match.group(1),
                    'original_message': message
                }
        
        # Keywords for updates in multiple languages
        update_keywords = [
            'update', 'change', 'modify', 'set', 'edit',
            'अपडेट', 'बदलें', 'परिवर्तन', 'सेट',
            'আপডেট', 'পরিবর্তন', 'সেট', 'সম্পাদনা'
        ]
        
        if not any(keyword in msg_lower for keyword in update_keywords):
            return None
        
        # Try to extract simulation ID from message
        sim_id_pattern = r'(?:simulation[:\s]+|sim[:\s]+|id[:\s]+|for\s+)([a-z0-9_-]+(?:_[a-z0-9_-]+)*)'
        sim_match = re.search(sim_id_pattern, message, re.IGNORECASE)
        simulation_id = sim_match.group(1) if sim_match else None
        
        # Try to extract field and value with more patterns
        field_name = None
        new_value = None
        
        # Pattern 1: "Patients Enrolled to 202"
        if re.search(r'patient(?:s)?\s+enrolled\s+to\s+(\d+)', msg_lower):
            match = re.search(r'patient(?:s)?\s+enrolled\s+to\s+(\d+)', msg_lower)
            field_name = 'patients_enrolled'
            new_value = match.group(1)
        # Pattern 2: "patient number to X"
        elif re.search(r'patient\s+(?:number|count)\s+to\s+(\d+)', msg_lower):
            match = re.search(r'patient\s+(?:number|count)\s+to\s+(\d+)', msg_lower)
            field_name = 'patients_enrolled'
            new_value = match.group(1)
        # Pattern 3: "dropout rate to X"
        elif re.search(r'dropout\s+rate\s+to\s+([\d.]+)', msg_lower):
            match = re.search(r'dropout\s+rate\s+to\s+([\d.]+)', msg_lower)
            field_name = 'dropout_rate'
            new_value = match.group(1)
        # Pattern 4: Generic "field to value"
        else:
            field_value_patterns = [
                r'(\w+)\s+to\s+([\d.]+)',
                r'(\w+)\s*:\s*([\d.]+)',
                r'(\w+)\s*=\s*([\d.]+)',
            ]
            
            for pattern in field_value_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    field_name = match.group(1)
                    new_value = match.group(2)
                    break
        
        if field_name:
            return {
                'simulation_id': simulation_id or self.current_simulation_id,
                'field': field_name,
                'new_value': new_value,
                'original_message': message
            }
        
        return None
    
    def _parse_translation_intent(self, message: str) -> Optional[Dict]:
        """Parse message for translation request."""
        
        msg_lower = message.lower()
        
        # Translation keywords
        translate_keywords = [
            'translate', 'translation', 'convert to',
            'अनुवाद', 'में बदलें',
            'অনুবাদ', 'রূপান্তর'
        ]
        
        if not any(keyword in msg_lower for keyword in translate_keywords):
            return None
        
        # Extract target language
        language_map = {
            'hindi': 'hi', 'हिंदी': 'hi', 'हिन्दी': 'hi',
            'bengali': 'bn', 'bangla': 'bn', 'বাংলা': 'bn',
            'english': 'en', 'अंग्रेजी': 'en', 'ইংরেজি': 'en'
        }
        
        target_language = None
        for lang_name, lang_code in language_map.items():
            if lang_name in msg_lower:
                target_language = lang_code
                break
        
        return {
            'target_language': target_language,
            'original_message': message
        }
    
    async def _handle_chat(self, message: str) -> Dict[str, Any]:
        """Handle regular chat conversation."""
        
        try:
            # Get user's simulations for context
            user_simulations = self.db.get_user_simulations(self.user_id)
            
            # Build conversation context
            system_prompt = self._get_system_prompt(user_simulations)
            
            # Add conversation history
            chat_history = self.conversation_history[-5:]  # Last 5 messages for context
            
            # Create prompt with history
            full_prompt = f"{system_prompt}\n\n**Conversation History:**\n"
            for msg in chat_history:
                full_prompt += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
            full_prompt += f"**Current User Message:**\n{message}\n\n**Your Response:**"
            
            # Generate response with Gemini (with automatic fallback)
            logger.info(f"Calling Gemini API for user {self.username}")
            assistant_response = self._call_gemini_with_fallback(full_prompt)
            logger.info(f"Gemini API response received: {len(assistant_response)} chars")
            
            # Store in conversation history
            self.conversation_history.append({
                'user': message,
                'assistant': assistant_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save to database
            self.db.save_chat_message(
                user_id=self.user_id,
                message=message,
                response=assistant_response,
                language=self.language,
                action_type='chat'
            )
            
            logger.info(f"Chat response generated for {self.username}")
            
            return {
                'response': assistant_response,
                'language': self.language,
                'action_type': 'chat',
                'action_data': None
            }
            
        except Exception as e:
            logger.error(f"Error in _handle_chat: {str(e)}", exc_info=True)
            return {
                'response': f"I encountered an error: {str(e)}. Please make sure your Gemini API key is configured correctly.",
                'language': self.language or 'en',
                'action_type': 'error',
                'action_data': None
            }
    
    async def _handle_update_request(self, intent: Dict) -> Dict[str, Any]:
        """Handle simulation update request."""
        
        simulation_id = intent['simulation_id']
        field = intent['field']
        new_value = intent['new_value']
        
        # If no simulation ID, try to use current simulation or ask user to specify
        if not simulation_id:
            if self.current_simulation_id:
                # Use the simulation currently being discussed
                simulation_id = self.current_simulation_id
                logger.info(f"Using current simulation context: {simulation_id}")
            else:
                # Check if user has only one simulation
                user_sims = self.db.get_user_simulations(self.user_id)
                if len(user_sims) == 1:
                    simulation_id = user_sims[0]['simulation_id']
                else:
                    sim_list = "\n".join([f"- {s['simulation_id']}" for s in user_sims])
                    return {
                        'response': f"Please specify which simulation to update:\n{sim_list}",
                        'language': self.language,
                        'action_type': 'clarification',
                        'action_data': None
                    }
        
        # Verify ownership
        if not self.db.verify_simulation_owner(simulation_id, self.user_id):
            return {
                'response': "❌ You don't have permission to update this simulation.",
                'language': self.language,
                'action_type': 'error',
                'action_data': None
            }
        
        # Get current simulation data
        simulation = self.db.get_simulation_by_id(simulation_id)
        if not simulation:
            return {
                'response': f"❌ Simulation {simulation_id} not found.",
                'language': self.language,
                'action_type': 'error',
                'action_data': None
            }
        
        # Get current value
        current_value = simulation.get(field, 'N/A')
        
        # Store pending action
        self.pending_action = {
            'type': 'update',
            'simulation_id': simulation_id,
            'field': field,
            'current_value': current_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate confirmation message in user's language
        confirmation_msg = self._generate_confirmation_message(
            simulation_id, field, current_value, new_value
        )
        
        logger.info(f"Update request pending confirmation for {self.username}")
        
        return {
            'response': confirmation_msg,
            'language': self.language,
            'action_type': 'update_request',
            'action_data': self.pending_action
        }
    
    def _generate_confirmation_message(self, sim_id: str, field: str, old_val: Any, new_val: Any) -> str:
        """Generate confirmation message in appropriate language."""
        
        if self.language == 'hi':
            return f"""📝 अपडेट अनुरोध:
- सिमुलेशन ID: {sim_id}
- फ़ील्ड: {field}
- वर्तमान मान: {old_val}
- नया मान: {new_val}

जारी रखने के लिए 'हां' या 'YES' का जवाब दें, या रद्द करने के लिए 'नहीं' या 'NO' का जवाब दें।"""
        
        elif self.language == 'bn':
            return f"""📝 আপডেট অনুরোধ:
- সিমুলেশন ID: {sim_id}
- ফিল্ড: {field}
- বর্তমান মান: {old_val}
- নতুন মান: {new_val}

এগিয়ে যেতে 'হ্যাঁ' বা 'YES' উত্তর দিন, অথবা বাতিল করতে 'না' বা 'NO' উত্তর দিন।"""
        
        else:  # English
            return f"""📝 Update Request Detected:
- Simulation ID: {sim_id}
- Field: {field}
- Current Value: {old_val}
- New Value: {new_val}

Reply with 'YES' or 'CONFIRM' to proceed, or 'NO' or 'CANCEL' to abort."""
    
    async def _handle_confirmation(self, message: str) -> Dict[str, Any]:
        """Handle user confirmation for pending action."""
        
        msg_lower = message.lower().strip()
        
        # Check if confirmed
        yes_keywords = ['yes', 'confirm', 'ok', 'y', 'हां', 'हाँ', 'जी', 'হ্যাঁ', 'ঠিক']
        no_keywords = ['no', 'cancel', 'n', 'नहीं', 'না']
        
        is_confirmed = any(keyword in msg_lower for keyword in yes_keywords)
        is_cancelled = any(keyword in msg_lower for keyword in no_keywords)
        
        if not (is_confirmed or is_cancelled):
            return {
                'response': "Please reply with YES to confirm or NO to cancel.",
                'language': self.language,
                'action_type': 'clarification',
                'action_data': None
            }
        
        if is_cancelled:
            self.pending_action = None
            return {
                'response': "❌ Update cancelled.",
                'language': self.language,
                'action_type': 'cancelled',
                'action_data': None
            }
        
        # Execute the update
        if self.pending_action['type'] == 'update':
            return await self._execute_update()
        
        return {
            'response': "No pending action to confirm.",
            'language': self.language,
            'action_type': 'error',
            'action_data': None
        }
    
    async def _execute_update(self) -> Dict[str, Any]:
        """Execute the pending update."""
        
        try:
            action = self.pending_action
            simulation_id = action['simulation_id']
            field = action['field']
            old_value = action['current_value']
            new_value = action['new_value']
            
            # Update the simulation data
            success = self.db.update_simulation_data(
                simulation_id=simulation_id,
                field=field,
                new_value=new_value
            )
            
            if not success:
                return {
                    'response': f"❌ Failed to update {field}. Please try again.",
                    'language': self.language,
                    'action_type': 'error',
                    'action_data': None
                }
            
            # Save to chat history
            self.db.save_chat_message(
                user_id=self.user_id,
                message=f"Confirmed update: {field} = {new_value}",
                response=f"Updated {field} from {old_value} to {new_value}",
                language=self.language,
                action_type='update_executed',
                simulation_id=simulation_id
            )
            
            # Clear pending action
            self.pending_action = None
            
            # Generate success message with old and new values
            success_msg = self._generate_success_message(simulation_id, field, old_value, new_value)
            
            logger.info(f"Update executed successfully: {simulation_id} - {field}: {old_value} → {new_value}")
            
            return {
                'response': success_msg,
                'language': self.language,
                'action_type': 'update_executed',
                'action_data': {
                    'simulation_id': simulation_id,
                    'field': field,
                    'old_value': old_value,
                    'new_value': new_value,
                    'resimulation_required': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing update: {str(e)}", exc_info=True)
            return {
                'response': f"❌ Error: {str(e)}",
                'language': self.language,
                'action_type': 'error',
                'action_data': None
            }
    
    def _generate_success_message(self, sim_id: str, field: str, old_val: Any, new_val: Any) -> str:
        """Generate success message in appropriate language."""
        
        if self.language == 'hi':
            return f"""✅ सफलतापूर्वक अपडेट किया गया!

- सिमुलेशन ID: {sim_id}
- फ़ील्ड: {field}
- पुराना मान: {old_val}
- नया मान: {new_val}

🔄 सिमुलेशन को फिर से प्रोसेस किया जा रहा है...
परिणाम कुछ मिनटों में तैयार हो जाएंगे।"""
        
        elif self.language == 'bn':
            return f"""✅ সফলভাবে আপডেট করা হয়েছে!

- সিমুলেশন ID: {sim_id}
- ফিল্ড: {field}
- পুরাতন মান: {old_val}
- নতুন মান: {new_val}

🔄 সিমুলেশন পুনরায় প্রক্রিয়া করা হচ্ছে...
ফলাফল কয়েক মিনিটের মধ্যে প্রস্তুত হবে।"""
        
        else:  # English
            return f"""✅ Successfully Updated!

- Simulation ID: {sim_id}
- Field: {field}
- Previous Value: {old_val}
- New Value: {new_val}

🔄 Re-processing simulation...
Results will be ready in a few minutes."""
    
    async def _handle_translation_request(self, intent: Dict) -> Dict[str, Any]:
        """Handle translation request."""
        
        target_language = intent['target_language']
        
        if not target_language:
            return {
                'response': "Please specify target language (English, Hindi, or Bengali).",
                'language': self.language,
                'action_type': 'clarification',
                'action_data': None
            }
        
        # Use Gemini to translate (with automatic fallback)
        prompt = f"""Translate the following to {target_language}:
{intent['original_message']}

Provide translation only, no explanations."""
        
        translation = self._call_gemini_with_fallback(prompt)
        
        return {
            'response': f"Translation:\n{translation}",
            'language': target_language,
            'action_type': 'translation',
            'action_data': {'translation': translation}
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history."""
        return self.conversation_history
    
    def clear_pending_action(self):
        """Clear any pending action (timeout or user request)."""
        self.pending_action = None
