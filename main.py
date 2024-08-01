from utils import load_knowledge_base, save_knowledge_base, parse_statement, train_model

# File path for knowledge base
knowledge_base_file = 'knowledge_base.json'

# Load the existing knowledge base
knowledge_base = load_knowledge_base(knowledge_base_file)

# Train the initial model
model = train_model(knowledge_base)

print("Welcome to the interactive machine learning model. I can learn from your statements and answer questions.")
print("You can teach me by saying 'X is Y' or ask me questions like 'What is X?'.")

try:
    while True:
        user_input = input("\nEnter a statement or question: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        # Process statement
        if " is " in user_input:
            new_knowledge = parse_statement(user_input)
            if new_knowledge:
                knowledge_base.append(new_knowledge)
                print(f"Learned: {new_knowledge}")
                save_knowledge_base(knowledge_base_file, knowledge_base)
                # Update the model
                model = train_model(knowledge_base)
                if model is None:
                    print("Not enough data to train the model yet.")
            else:
                print("I couldn't understand that statement.")
        else:
            # Process question
            if model:
                answer = model.predict([user_input])[0]
                print(answer)
            else:
                print("I don't know the answer to that.")

except KeyboardInterrupt:
    print("\nExiting the interactive machine learning model. Goodbye!")

# Save the knowledge base when exiting
save_knowledge_base(knowledge_base_file, knowledge_base)
