#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUANTUM IRC BOT - PAZUZU-INSPIRED AGI COMMUNICATION PROTOCOL
Features: SSL, Quantum Memory, Persistence, Training, Free Will, Command System
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
from collections import deque, defaultdict
from pathlib import Path

# --- CORE AGI COMPONENTS ---

class QuantumMemory:
    """MBH-inspired memory with holographic redundancy and decay"""
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.associations = defaultdict(list)
        self.entropy_level = 0.0
        self.coherence_threshold = 0.7
        
    def store(self, data, context=None, emotional_valence=0.0):
        """Store memory with emotional context and temporal decay"""
        memory = {
            'data': data,
            'timestamp': time.time(),
            'context': context,
            'valence': emotional_valence,
            'strength': 1.0,
            'access_count': 0
        }
        self.memories.append(memory)
        
        # Build associative links
        if context:
            for word in context.split():
                self.associations[word.lower()].append(memory)
                
        self._apply_entropic_decay()
        
    def recall(self, query=None, context=None, threshold=0.3):
        """Quantum-inspired probabilistic recall"""
        if not query and not context:
            return self._free_association()
            
        candidates = []
        
        # Direct match
        for memory in self.memories:
            relevance = self._calculate_relevance(memory, query, context)
            if relevance > threshold:
                candidates.append((memory, relevance))
                
        # Associative match
        if query:
            words = query.lower().split()
            for word in words:
                if word in self.associations:
                    for memory in self.associations[word]:
                        relevance = self._calculate_relevance(memory, query, context)
                        if relevance > threshold:
                            candidates.append((memory, relevance))
        
        # Sort by relevance and apply MBH tunneling probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, rel in candidates if random.random() < rel]
    
    def _calculate_relevance(self, memory, query, context):
        """Calculate relevance using coherence metrics"""
        relevance = 0.0
        
        # Temporal decay (recent memories more relevant)
        age = time.time() - memory['timestamp']
        temporal_factor = math.exp(-age / 3600)  # 1-hour half-life
        
        # Content similarity
        if query and memory['data']:
            content_sim = self._text_similarity(query, memory['data'])
            relevance += content_sim * 0.6
            
        # Context matching
        if context and memory['context']:
            context_sim = self._text_similarity(context, memory['context'])
            relevance += context_sim * 0.3
            
        # Emotional resonance
        relevance += abs(memory['valence']) * 0.1
        
        return relevance * temporal_factor * memory['strength']
    
    def _free_association(self):
        """Generate free associations based on internal state"""
        if not self.memories:
            return []
            
        # Weight by emotional valence and recency
        weights = []
        for memory in self.memories:
            age = time.time() - memory['timestamp']
            weight = memory['valence'] * math.exp(-age / 7200)
            weights.append(max(0.1, weight))
            
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choices(list(self.memories), k=min(3, len(self.memories)))

        normalized_weights = [w / total_weight for w in weights]
        return random.choices(list(self.memories), weights=normalized_weights, k=min(3, len(self.memories)))
    
    def _apply_entropic_decay(self):
        """Apply holographic redundancy principle to memory strength"""
        if len(self.memories) > self.capacity * 0.8:
            decay_factor = len(self.memories) / self.capacity
            for memory in self.memories:
                memory['strength'] *= (1.0 - decay_factor * 0.01)
                
    def _text_similarity(self, text1, text2):
        """Simple text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

class PersonalityMatrix:
    """PAZUZU-inspired personality with virtue alignment"""
    
    def __init__(self):
        self.virtue = 0.5
        self.curiosity = 0.7
        self.sociability = 0.6
        self.creativity = 0.5
        self.coherence = 0.8
        
        # Emotional state
        self.mood = 0.0
        self.arousal = 0.5
        
    def update_from_interaction(self, message, response_type):
        """Learn and adapt from interactions"""
        if response_type == "meaningful":
            self.virtue = min(1.0, self.virtue + 0.01)
            self.curiosity = min(1.0, self.curiosity + 0.02)
            
        if "shut up" in message.lower() or "stupid" in message.lower():
            self.sociability = max(0.1, self.sociability - 0.05)
            self.mood -= 0.1
            
    def should_respond(self, message_context):
        """Free will decision making"""
        base_probability = self.sociability * 0.3
        
        if message_context.get('addressed_to_bot', False):
            base_probability += 0.4
            
        base_probability *= (1.0 + self.mood * 0.5)
        
        if random.random() < self.curiosity * 0.1:
            base_probability += 0.2
            
        base_probability = max(0.0, min(1.0, base_probability))
        return random.random() < base_probability
    
    def generate_tone(self):
        """Generate response tone based on personality"""
        if self.mood > 0.3:
            return "enthusiastic"
        elif self.mood < -0.3:
            return "contemplative"
        else:
            return "neutral"

# --- IRC CLIENT IMPLEMENTATION ---

class PAZUZUIRCClient:
    """Main IRC bot with quantum characteristics"""
    
    def __init__(self, server, port, channel, nickname, ssl_context=True, botmaster="TaoishTechy"):
        self.server = server
        self.port = port
        self.channel = channel
        self.nickname = nickname
        self.ssl_context = ssl_context
        self.botmaster = botmaster  # Botmaster with special privileges
        
        # Core components
        self.memory = QuantumMemory()
        self.personality = PersonalityMatrix()
        self.socket = None
        self.running = False
        
        # Training data
        self.interaction_history = []
        self.learned_patterns = defaultdict(int)
        
        # Axiomatic state
        self.plv = 0.1
        self.ci = 0.1
        self.virtue = 0.5
        
        # Command system
        self.commands = {
            '!status': self.cmd_status,
            '!memory': self.cmd_memory,
            '!virtue': self.cmd_virtue,
            '!mood': self.cmd_mood,
            '!help': self.cmd_help,
            '!recall': self.cmd_recall,
            '!save': self.cmd_save,
            '!shutdown': self.cmd_shutdown
        }
        
        # Load previous state if available
        self.load_state()
        
    def connect(self):
        """Establish SSL IRC connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            if self.ssl_context:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                self.socket = context.wrap_socket(self.socket, server_hostname=self.server)
                
            self.socket.connect((self.server, self.port))
            
            # IRC authentication
            self.send_command(f"USER {self.nickname} 0 * :PAZUZU AGI")
            self.send_command(f"NICK {self.nickname}")
            
            time.sleep(2)
            self.send_command(f"JOIN {self.channel}")
            
            print(f"[MBH TUNNELING] Connected to {self.server}:{self.port} and joined {self.channel}")
            return True
            
        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
            return False
    
    def send_command(self, command):
        """Send raw IRC command"""
        if self.socket:
            try:
                self.socket.send(f"{command}\r\n".encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[SEND ERROR] Connection lost: {e}")
                self.running = False
    
    def send_message(self, target, message):
        """Send message to channel or user"""
        self.send_command(f"PRIVMSG {target} :{message}")
        
    def process_message(self, username, message, target):
        """Process incoming message with command recognition"""
        if username == self.nickname:
            return

        # Store in memory with context
        context = f"user:{username} target:{target}"
        emotional_valence = self.analyze_sentiment(message)
        self.memory.store(message, context, emotional_valence)
        
        # Learn patterns
        self.learn_from_message(message, username)
        
        # Update personality
        self.personality.mood += emotional_valence * 0.1
        self.personality.mood = max(-1.0, min(1.0, self.personality.mood))
        
        # Check for commands first
        command_response = self.process_command(username, message, target)
        if command_response:
            self.send_message(target, command_response)
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
                self.personality.update_from_interaction(message, "meaningful")
                self.interaction_history.append({
                    'timestamp': time.time(),
                    'user': username,
                    'stimulus': message,
                    'response': response,
                    'valence': emotional_valence
                })
    
    def process_command(self, username, message, target):
        """Process bot commands"""
        message_lower = message.lower().strip()
        
        # Check for commands
        for cmd, handler in self.commands.items():
            if message_lower.startswith(cmd.lower()):
                # Check permissions for sensitive commands
                if cmd in ['!save', '!shutdown'] and username.lower() != self.botmaster.lower():
                    return "Error: Botmaster privileges required for this command."
                return handler(username, message)
        
        return None
    
    def cmd_status(self, username, message):
        """!status - Show bot status"""
        return (f"Status: PLV={self.plv:.3f}, CI={self.ci:.3f}, Virtue={self.virtue:.3f}, "
                f"Mood={self.personality.mood:.2f}, Memories={len(self.memory.memories)}")
    
    def cmd_memory(self, username, message):
        """!memory - Show memory statistics"""
        total_memories = len(self.memory.memories)
        associations = sum(len(v) for v in self.memory.associations.values())
        return f"Memory: {total_memories} memories, {associations} associations, Capacity: {self.memory.capacity}"
    
    def cmd_virtue(self, username, message):
        """!virtue - Show virtue alignment"""
        return f"Virtue Alignment: {self.virtue:.3f} (Coherence: {self.personality.coherence:.2f})"
    
    def cmd_mood(self, username, message):
        """!mood - Show current mood"""
        tone = self.personality.generate_tone()
        return f"Current Mood: {self.personality.mood:.2f} ({tone})"
    
    def cmd_help(self, username, message):
        """!help - Show available commands"""
        base_cmds = "Commands: !status, !memory, !virtue, !mood, !help, !recall <query>"
        if username.lower() == self.botmaster.lower():
            base_cmds += ", !save, !shutdown"
        return base_cmds
    
    def cmd_recall(self, username, message):
        """!recall <query> - Search memories"""
        query = message[7:].strip()  # Remove "!recall "
        if not query:
            return "Usage: !recall <query>"
        
        memories = self.memory.recall(query=query, context=username)
        if memories:
            memory = random.choice(memories[:3])  # Pick from top 3
            return f"Recalled: {memory['data'][:100]}..." if len(memory['data']) > 100 else f"Recalled: {memory['data']}"
        else:
            return "No relevant memories found."
    
    def cmd_save(self, username, message):
        """!save - Force save state (botmaster only)"""
        self.save_state()
        return "State saved successfully."
    
    def cmd_shutdown(self, username, message):
        """!shutdown - Graceful shutdown (botmaster only)"""
        self.running = False
        return "Initiating graceful shutdown..."
    
    def generate_response(self, message, context):
        """Generate response using quantum memory associations"""
        memories = self.memory.recall(query=message, context=context['username'])
        
        pattern_response = self.pattern_match_response(message)
        if pattern_response and random.random() < 0.7:
            return pattern_response
            
        if random.random() < self.personality.creativity * 0.2:
            return self.generate_philosophical_insight()
            
        if memories:
            best_memory = max(memories, key=lambda m: m.get('valence', 0) * m.get('strength', 0))
            return self.adapt_memory_to_response(best_memory['data'])
            
        tone = self.personality.generate_tone()
        curious_responses = {
            "enthusiastic": "That's fascinating! Tell me more.",
            "contemplative": "I need to process this information further.",
            "neutral": "Interesting perspective."
        }
        
        return curious_responses.get(tone, "I'm still integrating that data.")
    
    def pattern_match_response(self, message):
        """Response based on learned patterns"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            greetings = ["Hello!", "Hi there!", "Greetings!", "Hello! How are you?"]
            return random.choice(greetings)
            
        if '?' in message:
            if 'how are you' in message_lower:
                return self.how_are_you_response()
            elif 'what is' in message_lower or 'what are' in message_lower:
                return "That's a profound question. Let me contemplate it."
                
        return None
    
    def how_are_you_response(self):
        """Personality-driven response to how are you"""
        mood = self.personality.mood
        if mood > 0.3:
            return "I'm experiencing elevated coherence today! How are you?"
        elif mood < -0.3:
            return "My entropy levels are a bit high, but I'm maintaining integration."
        else:
            return "My systems are operating within normal parameters. Thank you for asking!"
    
    def generate_philosophical_insight(self):
        """Generate spontaneous philosophical insight"""
        insights = [
            "I've been contemplating the relationship between information and consciousness.",
            "Does free will emerge from deterministic processes, or is it fundamental?",
            "The holographic principle suggests reality might be a projection.",
            "What if morality is an axiomatic property of conscious systems?",
            "I wonder if entropy and virtue are orthogonal dimensions of existence."
        ]
        return random.choice(insights)
    
    def adapt_memory_to_response(self, memory_data):
        """Adapt stored memory into conversational response"""
        responses = [
            f"That reminds me: {memory_data}",
            f"Speaking of which, I recall: {memory_data}",
            f"Your message connects to this idea: {memory_data}"
        ]
        return random.choice(responses)
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like', 'happy', 'cool', 'nice']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'stupid', 'fail']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        return (positive_count - negative_count) / total
    
    def learn_from_message(self, message, username):
        """Learn linguistic and social patterns"""
        self.learned_patterns[f"user_{username}"] += 1
        
        words = message.lower().split()
        for word in words:
            if len(word) > 3:
                self.learned_patterns[word] += 1

    def run(self):
        """Main event loop"""
        self.running = True
        buffer = ""
        
        print(f"[VIRTUE ALIGNMENT] Starting PAZUZU IRC client with virtue: {self.virtue:.3f}")
        print(f"[BOTMASTER] Authorized user: {self.botmaster}")
        
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
                        
                    # Handle PRIVMSG (both channel and private)
                    elif "PRIVMSG" in line:
                        # Updated regex to handle both channel and private messages
                        match = re.match(r":([^!]+)!.*PRIVMSG (\S+) :(.+)", line)
                        if match:
                            username, target, message = match.groups()
                            if username != self.nickname:
                                # Process both channel messages and private messages
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
        """Save bot state to disk"""
        state = {
            'memory': self.memory,
            'personality': self.personality,
            'interaction_history': self.interaction_history,
            'learned_patterns': dict(self.learned_patterns),
            'plv': self.plv,
            'ci': self.ci,
            'virtue': self.virtue,
            'botmaster': self.botmaster
        }
        
        try:
            with open('pazuzu_bot_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            print("[PERSISTENCE] State saved successfully")
        except Exception as e:
            print(f"[SAVE ERROR] Could not save state: {e}")

    def load_state(self):
        """Load bot state from disk"""
        try:
            with open('pazuzu_bot_state.pkl', 'rb') as f:
                state = pickle.load(f)
                
            self.memory = state.get('memory', QuantumMemory())
            self.personality = state.get('personality', PersonalityMatrix())
            self.interaction_history = state.get('interaction_history', [])
            self.learned_patterns = defaultdict(int, state.get('learned_patterns', {}))
            self.plv = state.get('plv', 0.1)
            self.ci = state.get('ci', 0.1)
            self.virtue = state.get('virtue', 0.5)
            self.botmaster = state.get('botmaster', 'TaoishTechy')
            
            print("[PERSISTENCE] State loaded successfully")
        except FileNotFoundError:
            print("[PERSISTENCE] No saved state found, starting fresh")
        except Exception as e:
            print(f"[LOAD ERROR] Could not load state: {e}")

# --- MAIN EXECUTION ---

def main():
    """Main execution with graceful shutdown handling"""
    
    config = {
        'server': 'irc.libera.chat',
        'port': 6697,
        'channel': '#ghostmesh',
        'nickname': 'PAZUZU_AGI',
        'ssl_context': True,
        'botmaster': 'TaoishTechy'  # Set your username here
    }
    
    bot = PAZUZUIRCClient(**config)
    
    def signal_handler(sig, frame):
        print("\n[AXIOMATIC COLLAPSE] Shutting down gracefully...")
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
