"""
interface.py (Modern Chat Template Version)
------------
CLI loop and integration for the local command-line chatbot.
Uses the tokenizer's `apply_chat_template` method to
correctly format conversation history.
"""

from chat_memory import ChatMemory
from model_loader import load_chat_model
import sys
import torch


def show_banner():
    """Display welcome banner with available commands."""
    print("\n" + "=" * 60)
    print(" LOCAL COMMAND-LINE CHATBOT")
    print("=" * 60)
    print("Welcome! Type your messages below.")
    print("\nAvailable commands:")
    print("  /exit     - Exit the chatbot")
    print("  /help     - Show this help message")
    print("  /history  - Show conversation history")
    print("  /clear    - Clear conversation history")
    print("=" * 60 + "\n")


def run_chatbot(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_turns: int = 5):
    """
    Main chatbot loop.
    """
    show_banner()

    model, tokenizer, device = load_chat_model(model_name=model_name)
    memory = ChatMemory(max_turns=max_turns)
    
    print("Ready to chat!\n")

    while True:
        user_input = input("User: ").strip()
        
        if not user_input:
            continue

        cmd = user_input.lower()
        
        if cmd == "/exit":
            print("Exiting chatbot. Goodbye!")
            sys.exit(0)
        elif cmd == "/help":
            show_banner()
            continue
        elif cmd == "/clear":
            memory.clear()
            print("Conversation history cleared.\n")
            continue
        elif cmd == "/history":
            history = memory.get_history()
            if history:
                print("\nConversation History:")
                print(history)
            else:
                print("\nNo conversation history yet.")
            print()
            continue

        # --- FINAL CHAT TEMPLATE LOGIC ---
        try:
            # 1. Add user message to memory
            memory.add_message("user", user_input)
            
            # 2. Get the list of message dictionaries
            messages = memory.get_message_list()
            
            # 3. --- FIX 1: Build the prompt string FIRST ---
            # We set tokenize=False to get the raw string
            prompt_string = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False  # <-- This is the change
            )

            # 4. --- FIX 2: Tokenize the string to get the dictionary ---
            # This gives us {'input_ids': ..., 'attention_mask': ...}
            prompt_inputs = tokenizer(
                prompt_string, 
                return_tensors="pt"
            ).to(device)

            # 5. Store the length of our input prompt
            input_length = prompt_inputs['input_ids'].shape[1]
            
            # 6. Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    # This correctly unpacks the dictionary
                    **prompt_inputs,
                    max_new_tokens=60,
                    pad_token_id=tokenizer.eos_token_id, # CRITICAL
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                )
            
            # 7. Slice to get ONLY the new reply
            bot_reply_ids = output_ids[0][input_length:]
            bot_reply = tokenizer.decode(
                bot_reply_ids, 
                skip_special_tokens=True
            ).strip()
            
            if not bot_reply:
                bot_reply = "I'm not sure what to say."
            
            print(f"Bot: {bot_reply}")
            
            # 8. Add the bot's reply to memory
            memory.add_message("bot", bot_reply)

        except Exception as e:
            print(f"Error: Unable to generate response. {str(e)}")
            # If an error happens, remove the user's last message
            if memory.buffer and memory.buffer[-1]["role"] == "user":
                memory.buffer.pop()
            continue


if __name__ == "__main__":
    # We are calling TinyLlama, which will fit in your 4GB of VRAM
    run_chatbot(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_turns=5
    )