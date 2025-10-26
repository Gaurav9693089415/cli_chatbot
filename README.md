
-----

# Local Command-Line Chatbot (Hugging Face)

A fully functional **local chatbot interface** built in Python using the Hugging Face `transformers` library. It runs a small, modern language model locally to maintain coherent, multi-turn conversations without any APIs.


##  Features

  * **100% Local Inference:** Runs entirely on  machine.
  * **Auto-Detects Hardware:** Automatically uses your NVIDIA GPU (CUDA) for acceleration if available, otherwise falls back to CPU.
  * **Modern Chat Model:** Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, a small but powerful model that is both **factual** (knows capitals) and **conversational** (understands follow-up questions).
  * **Robust Context Management:** Uses the official `tokenizer.apply_chat_template` method to perfectly format conversational history, which is far more reliable than manual string building.
  * **Sliding Window Memory:** Remembers the last 5 turns of conversation using an efficient `deque` collection.
  * **Modular Code:** Organized as required by the assignment:
      * `model_loader.py`
      * `chat_memory.py`
      * `cli_interface.py` (your main file)
  * **CLI Commands:** Includes `/exit`, `/help`, `/history`, and `/clear` for full control.

-----

##  Installation & Setup

### 1\. Clone the Project

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2\. Create and Activate a Virtual Environment

```bash
# Create the environment
python -m venv myenv

# On Windows
myenv\Scripts\activate

# On macOS/Linux
source myenv/bin/activate
```

### 3\. Create `requirements.txt`

Create a file named `requirements.txt` and add the following libraries:

```txt
transformers>=4.35.0
torch>=1.13.0
sentencepiece
accelerate
```

### 4\. Install Dependencies

Install the required libraries from your `requirements.txt` file.

```bash
pip install -r requirements.txt
```

-----

##  How to Run

The script will **automatically detect and use your GPU** (CUDA) if one is available.

Simply run the main interface file:

```bash
python cli_interface.py
```

-----

##  Example Interaction

The bot successfully handles multi-turn, context-aware, factual questions.

```
============================================================
 LOCAL COMMAND-LINE CHATBOT
============================================================
Welcome! Type your messages below.

Available commands:
  /exit     - Exit the chatbot
  /help     - Show this help message
  /history  - Show conversation history
  /clear    - Clear conversation history
============================================================

Ready to chat!

User: What is the capital of France?
Bot: The capital of France is Paris.

User: And what about Italy?
Bot: The capital of Italy is Rome.

User: /exit
Exiting chatbot. Goodbye!
```

-----

##  Project Structure

```
your_project_directory/
├── model_loader.py       # Handles loading the model & tokenizer (Auto-detects GPU)
├── chat_memory.py        # Manages the sliding window memory using deque
├── cli_interface.py      # The main application script with the chat loop
├── requirements.txt      # Project dependencies
└── README.md             
```

-----

##  Design Decisions

  * **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was chosen as the ideal model. It is small enough to run on consumer hardware (like a GTX 1650 with 4GB VRAM) while being a modern, instruction-tuned chat model. This solves the trade-off between older models that were either factual *or* conversational, but rarely both.
  * **Context Management:** The `tokenizer.apply_chat_template` method is the core of the chat logic. This is the modern, official Hugging Face standard for formatting conversation history. It dynamically builds the correct prompt string (e.g., `<|user|>\n...<|assistant|>\n...`) for the specific model, which is why the bot can correctly infer the context of "And what about Italy?".
  * **Efficiency:** The model is loaded in `float16` (half-precision) on CUDA devices to reduce VRAM usage and speed up inference. The chat memory uses a `deque` with a fixed `maxlen` for an efficient sliding window.
