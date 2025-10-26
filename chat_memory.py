"""
Chat Memory Module (Chat Template Optimized)
--------------------------
Maintains a sliding window of recent conversation turns
as a list of dictionaries, ready for a tokenizer's
apply_chat_template method.
"""

from collections import deque
from typing import Deque, List, Dict


class ChatMemory:
    def __init__(self, max_turns: int = 5):
        """
        Initialize sliding memory (last N user–bot turns).
        """
        self.max_turns = max_turns
        # Store message dictionaries: {"role": "user", "content": "..."}
        self.buffer: Deque[Dict[str, str]] = deque(maxlen=self.max_turns * 2)

    def add_message(self, role: str, content: str):
        """
        Add a message from user or bot.
        
        Args:
            role: Either 'user' or 'assistant' (or 'bot')
            content: The message content
        """
        # We use "assistant" as it's the standard role for this model
        role = "assistant" if role.lower() == "bot" else "user"
        
        self.buffer.append({"role": role, "content": content.strip()})

    def get_message_list(self) -> List[Dict[str, str]]:
        """
        Return the full list of messages.
        """
        return list(self.buffer)

    def get_history(self) -> str | None:
        """
        Return full formatted conversation for display purposes.
        """
        if not self.buffer:
            return None
            
        # Re-map "assistant" to "Bot" for display
        display_list = []
        for msg in self.buffer:
            role = "Bot" if msg["role"] == "assistant" else "User"
            display_list.append(f"{role}: {msg['content']}")
        return "\n".join(display_list)

    def clear(self):
        """Clear the conversation memory."""
        self.buffer.clear()