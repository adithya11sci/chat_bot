
import sys
import os
import chatbot

# Mock state if needed, but chatbot.py loads state on import if files exist
if not chatbot._state["ready"]:
    chatbot._load_index_from_disk()

try:
    print("Testing 'which book has more pages'...")
    res = chatbot.answer_question("which book has more pages")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    print("\nTesting 'Which book is most popular among readers?'...")
    res = chatbot.answer_question("Which book is most popular among readers?")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
