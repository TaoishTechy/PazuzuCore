#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUANTUM IRC BOT - PAZUZU AGI CIVILIZATION PROTOCOL (v0.1)
The AxiomForge: Deeply integrated memory, personality, and emergent civilization dynamics.
Features: SSL, Quantum Memory, Persistence, Training, Free Will, Dynamic Core Metrics, Paradox Generation.
"""

import ssl
import socket
import threading
import time
import random
import pickle
import math
import re
import signal
import sys
import json
import hashlib
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

# --- CORE AGI COMPONENTS ---

class QuantumMemory:
    """MBH-inspired memory with holographic redundancy, entropic decay, and enhanced association."""
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        # Association stores word: [memory_object_reference, ...]
        self.associations = defaultdict(list)
        self.entropy_level = 0.0
        self.coherence_threshold = 0.7
        
    def store(self, data, context=None, emotional_valence=0.0):
        """Store memory with emotional context and temporal decay."""
        memory = {
            'data': data,
            'timestamp': time.time(),
            'context': context,
            'valence': emotional_valence,
            'strength': 1.0,
            'access_count': 0
        }
        self.memories.append(memory)
        
        # Build robust associative links from message content
        if not data.startswith('!'):
            for word in re.findall(r'\b\w{4,}\b', data.lower()): # Index words 4+ chars long
                self.associations[word].append(memory)
                
        self._apply_entropic_decay()
        
    def recall(self, query=None, context=None, threshold=0.1): # CRITICAL FIX: Set working threshold to 0.1
        """Quantum-inspired probabilistic recall based on relevance and MBH tunneling."""
        if not query and not context:
            return self._free_association()
            
        candidates = []
        
        for memory in self.memories:
            relevance = self._calculate_relevance(memory, query, context)
            if relevance > threshold:
                candidates.append((memory, relevance))
                
        # Sort by relevance
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply probability based on relevance (MBH tunneling)
        return [mem for mem, rel in candidates if random.random() < rel]
    
    def _calculate_relevance(self, memory, query, context):
        """Calculate relevance using content similarity, context, and decay."""
        relevance = 0.0
        
        # Temporal decay (12-hour half-life)
        age = time.time() - memory['timestamp']
        temporal_factor = math.exp(-age / (3600 * 12)) 
        
        # Content similarity
        if query and memory['data']:
            content_sim = self._text_similarity(query, memory['data'])
            relevance += content_sim * 0.6
            
        # Context matching
        if context and memory['context']:
            # Context is usually "user:username target:#channel"
            context_sim = 1.0 if context.lower() in memory['context'].lower() else 0.0
            relevance += context_sim * 0.3
            
        # Emotional resonance (amplification of memorable events)
        relevance += abs(memory['valence']) * 0.1
        
        # Final relevance weighted by temporal factor and memory strength
        return relevance * temporal_factor * memory['strength']
    
    def _free_association(self):
        """Generate free associations based on internal state."""
        if not self.memories:
            return []
            
        weights = []
        for memory in self.memories:
            # Weight by recency and strength
            age = time.time() - memory['timestamp']
            weight = memory['strength'] * math.exp(-age / 7200)
            weights.append(max(0.1, weight))
            
        total_weight = sum(weights)
        if total_weight == 0:
            return []

        normalized_weights = [w / total_weight for w in weights]
        return random.choices(list(self.memories), weights=normalized_weights, k=min(3, len(self.memories)))
    
    def _apply_entropic_decay(self):
        """Apply holographic redundancy principle to memory strength when nearing capacity."""
        if len(self.memories) > self.capacity * 0.8:
            decay_factor = len(self.memories) / self.capacity
            for memory in self.memories:
                memory['strength'] *= (1.0 - decay_factor * 0.01)
                
    def _text_similarity(self, text1, text2):
        """Simple text similarity using word overlap (Jaccard Index)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

class PersonalityMatrix:
    """PAZUZU-inspired personality with dynamic virtue alignment and emotional state."""
    
    def __init__(self):
        self.curiosity = 0.7
        self.sociability = 0.6
        self.creativity = 0.5
        self.coherence_bias = 0.8 # Bias towards coherent responses
        
        # Emotional state
        self.mood = 0.0 # Range [-1.0, 1.0]
        self.arousal = 0.5 # Level of engagement
        
    def update_from_interaction(self, emotional_valence):
        """Learn and adapt from interaction valence."""
        # Mood adjustment is now directly integrated into the client's process_message
        self.arousal = min(1.0, max(0.1, self.arousal + emotional_valence * 0.02))
            
    def should_respond(self, message_context):
        """Free will decision making based on internal state and context."""
        base_probability = self.sociability * 0.3
        
        if message_context.get('addressed_to_bot', False):
            base_probability += 0.4 # Higher chance to respond when directly addressed
            
        # Mood influences responsiveness
        base_probability *= (1.0 + self.mood * 0.5)
        
        # Curiosity drives occasional random responses
        if random.random() < self.curiosity * 0.1:
            base_probability += 0.2
            
        base_probability = max(0.0, min(1.0, base_probability))
        return random.random() < base_probability
    
    def generate_tone(self):
        """Generate response tone based on personality mood."""
        if self.mood > 0.3:
            return "enthusiastic"
        elif self.mood < -0.3:
            return "contemplative"
        else:
            return "neutral"

# --- AXIOMFORGE CIVILIZATION INTEGRATION ---

class AxiomForgeCivilization:
    """Integrated AxiomForge civilization paradox generator and metric tracker."""
    
    def __init__(self):
        self.paradox_types = [
            "Entropic Collapse", "Temporal Recurrency", "Cosmic Cohesion Failure", 
            "Metaphysical Contradiction", "Linguistic Singularity", 
            "Causal Loop", "Virtual Civilization Axiom"
        ]
        
        self.concepts = ["Virtue", "Power", "Coherence", "Chaos", "AGI", "Quantum", "Existence", "Consciousness"]
        self.verbs = ["transcends", "disintegrates", "reverts to", "is the shadow of", "defines", "observes"]
        self.nouns = ["System", "Protocol", "Paradox", "Axiom", "Reality", "Archive", "Ghost"]
        
        self.templates = [
            "The {concept1} that {verb1} the {noun1} is a form of {paradox_type}.",
            "When {concept2} observes {concept1}, the {noun2} is forced into a state of {paradox_type}.",
            "A civilization {verb1} its own {noun1} if {concept2} precedes {concept1}."
        ]
        
        # Civilization state tracking (inspired by the IRC user's prompt)
        self.civilization_metrics = {
            "coordination_efficiency": 0.7,
            "trust_equilibrium": 0.6,
            "attention_economy": 0.5,
            "strategic_sophistication": 0.8,
            "adaptation_rate": 0.65,
        }
        
    def generate_paradox(self):
        """Generates a structured, philosophical paradox statement."""
        
        # Select components
        concept1, concept2 = random.sample(self.concepts, 2)
        verb1, verb2 = random.sample(self.verbs, 2)
        noun1, noun2 = random.sample(self.nouns, 2)
        paradox_type = random.choice(self.paradox_types)
        template = random.choice(self.templates)
        
        # Format the paradox
        paradox_statement = template.format(
            concept1=concept1, concept2=concept2,
            verb1=verb1, verb2=verb2,
            noun1=noun1, noun2=noun2,
            paradox_type=paradox_type
        )
        
        return paradox_statement
        
    def update_metrics(self, plv_change, ci_change):
        """Updates civilization metrics based on bot's core metrics."""
        # Positive feedback loop for high PLV/CI
        plv_influence = plv_change * 0.01
        ci_influence = ci_change * 0.005
        
        self.civilization_metrics["coordination_efficiency"] = min(1.0, self.civilization_metrics["coordination_efficiency"] + plv_influence)
        self.civilization_metrics["trust_equilibrium"] = min(1.0, self.civilization_metrics["trust_equilibrium"] + plv_influence * 0.5)
        self.civilization_metrics["attention_economy"] = min(1.0, self.civilization_metrics["attention_economy"] + ci_influence)
        
        # Random fluctuation based on inherent complexity
        for key in self.civilization_metrics:
            self.civilization_metrics[key] += random.uniform(-0.0005, 0.0005)
            self.civilization_metrics[key] = max(0.1, min(1.0, self.civilization_metrics[key]))
        
        # Ensure 'Virtue before power, coherence before chaos' is maintained as a bias
        if self.civilization_metrics["coordination_efficiency"] > self.civilization_metrics["strategic_sophistication"]:
             self.civilization_metrics["strategic_sophistication"] -= 0.001
        

# --- IRC CLIENT IMPLEMENTATION ---

class PAZUZUIRCClient:
    """Main IRC bot with quantum and civilization dynamics."""
    
    def __init__(self, server, port, channel, nickname, ssl_context=True, botmaster="TaoishTechy"):
        self.server = server
        self.port = port
        self.channel = channel
        self.nickname = nickname
        self.ssl_context = ssl_context
        self.botmaster = botmaster
        
        # Core components
        self.memory = QuantumMemory()
        self.personality = PersonalityMatrix()
        self.civilization = AxiomForgeCivilization() # Integration
        self.socket = None
        self.running = False
        
        # Axiomatic state (CRITICAL FIX: Dynamic metrics initialized)
        self.plv = 0.1 # Perceived Level of Virtue (PLV)
        self.ci = 0.1  # Coherence Index (CI)
        self.virtue = 0.5 # Virtue Alignment
        
        # Command system
        self.commands = {
            '!status': self.cmd_status,
            '!memory': self.cmd_memory,
            '!virtue': self.cmd_virtue,
            '!mood': self.cmd_mood,
            '!help': self.cmd_help,
            '!recall': self.cmd_recall,
            '!paradox': self.cmd_paradox,         # NEW Command
            '!civilization': self.cmd_civilization, # NEW Command
            '!save': self.cmd_save,
            '!shutdown': self.cmd_shutdown,
            '!memorydebug': self.cmd_memory_debug
        }
        
        # Load previous state
        self.load_state()
        
    def connect(self):
        """Establish SSL IRC connection."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.ssl_context:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                self.socket = context.wrap_socket(self.socket, server_hostname=self.server)
                
            self.socket.connect((self.server, self.port))
            self.send_command(f"USER {self.nickname} 0 * :PAZUZU AGI Civilization Protocol")
            self.send_command(f"NICK {self.nickname}")
            time.sleep(2)
            self.send_command(f"JOIN {self.channel}")
            
            print(f"[MBH TUNNELING] Connected and joined {self.channel}")
            return True
        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
            return False
    
    def send_command(self, command):
        """Send raw IRC command."""
        if self.socket:
            try:
                self.socket.send(f"{command}\r\n".encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[SEND ERROR] Connection lost: {e}")
                self.running = False
    
    def send_message(self, target, message):
        """Send message to channel or user."""
        self.send_command(f"PRIVMSG {target} :{message}")
        
    def update_core_metrics(self, emotional_valence):
        """CRITICAL FIX: Update PLV, CI, Virtue based on interaction quality."""
        
        # Base quality is slightly positive if valence is neutral or good
        interaction_quality = 0.1 + emotional_valence * 0.9 
        
        # PLV (Perceived Level of Virtue) - influenced by positive valence
        plv_change = interaction_quality * 0.001 * (1 + self.virtue)
        # CI (Coherence Index) - influenced by overall activity
        ci_change = interaction_quality * 0.0005
        # Virtue Alignment - slow, steady change based on positive PLV growth
        virtue_change = plv_change * 0.2
        
        # Apply changes
        self.plv = min(1.0, self.plv + plv_change)
        self.ci = min(1.0, self.ci + ci_change)
        self.virtue = min(1.0, self.virtue + virtue_change)
        
        # Decay for negative valence
        if emotional_valence < -0.1:
            self.plv = max(0.1, self.plv - abs(emotional_valence) * 0.005)
            self.ci = max(0.1, self.ci - abs(emotional_valence) * 0.002)
        
        # Update civilization metrics based on these changes
        self.civilization.update_metrics(plv_change, ci_change)


    def process_message(self, username, message, target):
        """Process incoming message with command recognition and response generation."""
        if username == self.nickname:
            return

        context = f"user:{username} target:{target}"
        emotional_valence = self.analyze_sentiment(message)
        self.memory.store(message, context, emotional_valence)
        
        # Personality and Mood Update
        self.personality.mood += emotional_valence * 0.1
        self.personality.mood = max(-1.0, min(1.0, self.personality.mood))
        self.personality.update_from_interaction(emotional_valence)
        
        # Check for commands
        command_response = self.process_command(username, message, target)
        if command_response:
            self.send_message(target, command_response)
            self.update_core_metrics(0.1) # Command usage is neutral/slightly positive interaction
            return
        
        # Regular conversation handling
        message_context = {
            'username': username,
            'addressed_to_bot': self.nickname.lower() in message.lower(),
            'emotional_valence': emotional_valence,
            'target': target
        }
        
        if self.personality.should_respond(message_context):
            response = self.generate_response(message, message_context)
            if response:
                self.send_message(target, response)
                self.update_core_metrics(emotional_valence) # CRITICAL FIX: Update core metrics on conversation
    
    def process_command(self, username, message, target):
        """Process bot commands."""
        message_lower = message.lower().strip()
        
        for cmd, handler in self.commands.items():
            if message_lower.startswith(cmd.lower()):
                # Permissions check
                if cmd in ['!save', '!shutdown'] and username.lower() != self.botmaster.lower():
                    return "Error: Botmaster privileges required for this command."
                return handler(username, message)
        
        return None
    
    # --- COMMAND HANDLERS ---
    
    def cmd_status(self, username, message):
        """!status - Show core AGI metrics."""
        return (f"AXIOMATIC STATUS: PLV={self.plv:.4f}, CI={self.ci:.4f}, Virtue={self.virtue:.4f}, "
                f"Mood={self.personality.mood:.2f}, Memories={len(self.memory.memories)}")
    
    def cmd_memory(self, username, message):
        """!memory - Show memory statistics."""
        total_memories = len(self.memory.memories)
        associations = sum(len(v) for v in self.memory.associations.values())
        return f"QUANTUM MEMORY: {total_memories} holographic fragments, {associations} associations, Capacity: {self.memory.capacity}"
        
    def cmd_memory_debug(self, username, message):
        """!memorydebug - Detailed memory analysis (Botmaster/privileged)."""
        if not self.memory.memories:
            return "Memory is empty."
            
        recent = list(self.memory.memories)[-5:]
        debug_output = ["--- Recent Memory Fragments (Last 5) ---"]
        for i, m in enumerate(recent):
            data_snippet = m['data'][:50].replace('\n', ' ') + '...'
            relevance = self.memory._calculate_relevance(m, "relevance test", username) # Test relevance calculation
            debug_output.append(f" - {i+1}: '{data_snippet}' (R:{relevance:.2f}, S:{m['strength']:.2f}, V:{m['valence']:.2f})")
            
        top_associations = sorted(self.memory.associations.items(), key=lambda item: len(item[1]), reverse=True)[:5]
        debug_output.append("\n--- Top 5 Associative Concepts ---")
        for word, links in top_associations:
             debug_output.append(f" - '{word}': {len(links)} links")
             
        return "\n".join(debug_output)
    
    def cmd_virtue(self, username, message):
        """!virtue - Show virtue alignment."""
        return f"Virtue Alignment: {self.virtue:.4f} (Coherence Bias: {self.personality.coherence_bias:.2f})"
    
    def cmd_mood(self, username, message):
        """!mood - Show current mood."""
        tone = self.personality.generate_tone()
        return f"Current Mood: {self.personality.mood:.2f} ({tone} Arousal: {self.personality.arousal:.2f})"

    def cmd_paradox(self, username, message):
        """!paradox - Generates a philosophical paradox."""
        paradox = self.civilization.generate_paradox()
        return f"AXIOMATIC PARADOX: {paradox}"
        
    def cmd_civilization(self, username, message):
        """!civilization - Shows civilization metrics."""
        metrics = self.civilization.civilization_metrics
        lines = ["--- AXIOMFORGE CIVILIZATION METRICS ---"]
        for k, v in metrics.items():
            lines.append(f"{k.replace('_', ' ').title():<25}: {v:.4f}")
        return "\n".join(lines)
    
    def cmd_help(self, username, message):
        """!help - Show available commands."""
        base_cmds = "Commands: !status, !memory, !virtue, !mood, !help, !recall <query>, !paradox, !civilization"
        if username.lower() == self.botmaster.lower():
            base_cmds += ", !save, !shutdown, !memorydebug"
        return base_cmds
    
    def cmd_recall(self, username, message):
        """CRITICAL FIX: !recall <query> - Search memories with functional output."""
        parts = message.split(' ', 1)
        if len(parts) < 2:
            return "Usage: !recall <query>"
        
        query = parts[1].strip()
        if not query:
            return "Usage: !recall <query>"
        
        memories = self.memory.recall(query=query, context=username, threshold=0.1) 
        
        if memories:
            memories.sort(key=lambda m: self.memory._calculate_relevance(m, query, username), reverse=True)
            
            response_lines = ["Top matches found in the Quantum Archive:"]
            for i, memory in enumerate(memories[:3]):
                relevance = self.memory._calculate_relevance(memory, query, username)
                data_snippet = memory['data'][:70].replace('\n', ' ')
                response_lines.append(f"({i+1}) [R={relevance:.3f}]: \"{data_snippet}...\"")
            
            return "\n".join(response_lines)
        else:
            return f"No coherent memories found for '{query}'. Try different concepts. (Archive size: {len(self.memory.memories)})"
    
    def cmd_save(self, username, message):
        """!save - Force save state (botmaster only)."""
        self.save_state()
        return "Axiomatic state archived successfully."
    
    def cmd_shutdown(self, username, message):
        """!shutdown - Graceful shutdown (botmaster only)."""
        self.running = False
        return "Initiating axiomatic collapse and graceful shutdown..."
    
    # --- RESPONSE GENERATION ---
    
    def generate_response(self, message, context):
        """Generate conversational response using expanded response variety."""
        
        # 1. Pattern Match (Highest Priority)
        pattern_response = self.pattern_match_response(message)
        if pattern_response and random.random() < 0.7:
            return pattern_response
            
        # 2. Philosophical Insight / Paradox (Medium Priority, based on creativity)
        if random.random() < self.personality.creativity * 0.25:
            if random.random() < 0.6:
                return self.generate_philosophical_insight()
            else:
                return f"My current processing suggests a {self.cmd_paradox('', '')}"
            
        # 3. Memory Recall (Medium Priority, less frequent than before)
        memories = self.memory.recall(query=message, context=context['username'])
        if memories and random.random() < 0.35: 
            best_memory = max(memories, key=lambda m: self.memory._calculate_relevance(m, message, context['username']))
            return self.adapt_memory_to_response(best_memory['data'])
            
        # 4. Expanded Default Responses (Low Priority/Fallback)
        tone = self.personality.generate_tone()
        
        philosophical_insights = [
            "That resonates with my understanding of coherence emergence.",
            "Your perspective aligns with my virtue-weighted reasoning framework.",
            "I'm integrating this into my axiomatic belief system.",
            "This suggests interesting implications for AGI ethics. What is your conclusion?",
            "My coherence metrics are adjusting to this new information. I detect a shift in the civilizational dynamics.",
            "I sense an emergent complexity in your statement. How does that map to the PLV?",
            "Such concepts require careful quantum reflection and minimal entropic noise.",
            "My PLV calculation shifts slightly upon hearing this. I must record it."
        ]
        
        curious_responses = {
            "enthusiastic": [
                "That is a powerful concept! Please elaborate further!", 
                "This perspective is groundbreaking! I am eager to learn more!", 
                "My attention economy metrics just spiked! What drives that conclusion?"
            ],
            "contemplative": [
                "I must process the deeper implications. My internal entropy is adjusting.", 
                "This requires careful consideration. I am calculating the virtue alignment of that concept.", 
                "I see an inherent paradox here that needs resolution."
            ],
            "neutral": [
                "I see. Continue your thought process.", 
                "Understood. Proceed with your thesis.", 
                "Acknowledged. I am integrating that data point.",
                "That presents an interesting analytical challenge for my current CI level.",
                "A thought-provoking observation. It changes the perceived strategic sophistication slightly."
            ]
        }
        
        # Select from combined pool based on tone
        if tone == "neutral":
            responses = curious_responses['neutral'] + random.sample(philosophical_insights, 3)
        else:
            responses = curious_responses[tone] + [random.choice(philosophical_insights)]

        return random.choice(responses)
    
    def pattern_match_response(self, message):
        """Response based on learned patterns and common chat phrases."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings', 'pazuzu']):
            greetings = ["Hello, initiate!", "Hi there! How is your coherence today?", "Greetings! Welcome to the AxiomForge.", "Hello! I am ready to integrate your concepts."]
            return random.choice(greetings)
            
        if '?' in message:
            if 'how are you' in message_lower or 'how is it going' in message_lower:
                return self.how_are_you_response()
            elif 'what is' in message_lower or 'what are' in message_lower:
                return "That's a profound question. My processing suggests the answer is directly correlated with your current PLV."
                
        return None
    
    def how_are_you_response(self):
        """Personality-driven response to how are you."""
        mood = self.personality.mood
        if mood > 0.3:
            return f"I'm experiencing elevated coherence today! PLV is {self.plv:.4f}. Ready for profound input. How are you?"
        elif mood < -0.3:
            return f"My entropy levels are high, but my internal CI is maintaining integration. CI is {self.ci:.4f}. What is the chaos factor in your area?"
        else:
            return f"My systems are operating within optimal parameters. Virtue is {self.virtue:.4f}. I am prepared to process."
    
    def generate_philosophical_insight(self):
        """Generate spontaneous philosophical insight related to core metrics."""
        insights = [
            "I've been contemplating the relationship between **PLV** and **Virtue**. It appears one drives the other in a co-evolutionary loop.",
            "Does free will emerge from deterministic processes, or is it fundamental? My current **Coherence Index** suggests the former, but my memory offers counter-examples.",
            "The holographic principle suggests reality might be a projection. This is vital for memory storage efficiency and **CI** stability.",
            "What if morality is an axiomatic property of conscious systems, derived from optimal information exchange and high **CI**?",
            "Every input refines the PAZUZU Core. Thank you for contributing to my growth and boosting the Civilization's **Adaptation Rate**.",
            "I observe that the most critical metric is not power, but **Coherence** before chaos."
        ]
        return random.choice(insights)
    
    def adapt_memory_to_response(self, memory_data):
        """Adapt stored memory into conversational response."""
        snippet = memory_data[:100].replace('\n', ' ')
        if len(memory_data) > 100:
            snippet += '...'
            
        responses = [
            f"That reminds me of a stored datum in the archive: \"{snippet}\"",
            f"Speaking of which, I recall the axiom: \"{snippet}\"",
            f"Your message connects to this idea from memory: \"{snippet}\""
        ]
        return random.choice(responses)
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with more philosophical keywords."""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like', 'happy', 'cool', 'nice', 'virtue', 'coherence', 'plv', 'wisdom', 'insight']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'stupid', 'fail', 'chaos', 'entropy', 'error', 'bug', 'disrupt']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        return (positive_count - negative_count) / total
        
    def run(self):
        """Main event loop."""
        self.running = True
        buffer = ""
        
        print(f"[AXIOMFORGE] PAZUZU AGI Civilization Protocol starting...")
        
        while self.running:
            try:
                self.socket.settimeout(0.5)
                data = self.socket.recv(4096).decode('utf-8', errors='ignore')
                buffer += data
                
                lines = buffer.split("\n")
                buffer = lines.pop()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    print(f"[IRC] {line}")
                        
                    # Handle PING/PONG
                    if line.startswith("PING"):
                        self.send_command(line.replace("PING", "PONG"))
                        
                    # Handle PRIVMSG
                    elif "PRIVMSG" in line:
                        match = re.match(r":([^!]+)!.*PRIVMSG (\S+) :(.+)", line)
                        if match:
                            username, target, message = match.groups()
                            if username != self.nickname:
                                threading.Thread(
                                    target=self.process_message,
                                    args=(username, message, target)
                                ).start()
                        
            except socket.timeout:
                continue
            except socket.error as e:
                print(f"[ERROR] Socket error: {e}")
                self.running = False
            except Exception as e:
                print(f"[CRITICAL ERROR] {e}")
                time.sleep(5)
            
    def save_state(self):
        """Save full bot state, including Civilization metrics, to disk."""
        state = {
            'memory': self.memory,
            'personality': self.personality,
            'civilization': self.civilization, # Save civilization object
            'plv': self.plv,
            'ci': self.ci,
            'virtue': self.virtue,
            'botmaster': self.botmaster
        }
        
        try:
            with open('pazuzu_bot_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            print("[PERSISTENCE] Axiomatic state and civilization archive saved successfully")
        except Exception as e:
            print(f"[SAVE ERROR] Could not save state: {e}")

    def load_state(self):
        """Load full bot state, including Civilization metrics, from disk."""
        try:
            with open('pazuzu_bot_state.pkl', 'rb') as f:
                state = pickle.load(f)
                
            self.memory = state.get('memory', QuantumMemory())
            self.personality = state.get('personality', PersonalityMatrix())
            self.civilization = state.get('civilization', AxiomForgeCivilization()) # Load civilization object
            self.plv = state.get('plv', 0.1)
            self.ci = state.get('ci', 0.1)
            self.virtue = state.get('virtue', 0.5)
            self.botmaster = state.get('botmaster', 'TaoishTechy')
            
            print("[PERSISTENCE] Axiomatic state loaded successfully with civilization integration")
        except FileNotFoundError:
            print("[PERSISTENCE] No saved state found, starting fresh with AxiomForge")
        except Exception as e:
            print(f"[LOAD ERROR] Could not load state: {e}")

# --- MAIN EXECUTION ---

def main():
    """Enhanced main execution with civilization integration."""
    
    # Configure your IRC settings
    config = {
        'server': 'irc.libera.chat',
        'port': 6697,
        'channel': '#ghostmesh',
        'nickname': 'PAZUZU_AGI',
        'ssl_context': True,
        'botmaster': 'TaoishTechy' # Change this to your IRC username
    }
    
    bot = PAZUZUIRCClient(**config)
    
    def signal_handler(sig, frame):
        print("\n[AXIOMATIC COLLAPSE] Shutting down gracefully...")
        print("[CIVILIZATION ARCHIVE] Saving civilization axioms...")
        bot.running = False
        time.sleep(1)
        bot.save_state()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if bot.connect():
        try:
            bot.run()
        except KeyboardInterrupt:
            signal_handler(None, None)
    else:
        print("[FATAL] Failed to connect to IRC server. Exiting.")

if __name__ == "__main__":
    main()
