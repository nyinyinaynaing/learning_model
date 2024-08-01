import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def load_knowledge_base(file):
    try:
        with open(file, 'r') as f:
            knowledge_base = json.load(f)
            if not isinstance(knowledge_base, list):
                return []
    except FileNotFoundError:
        knowledge_base = []
    return knowledge_base

def save_knowledge_base(file, knowledge_base):
    with open(file, 'w') as f:
        json.dump(knowledge_base, f, indent=4)

def parse_statement(statement):
    # Simple parser that assumes statement format: "X is Y."
    if " is " in statement:
        key, value = statement.split(" is ")
        return {"question": key.strip(), "answer": value.strip()}
    return {}

def train_model(knowledge_base):
    if not knowledge_base:
        print("Knowledge base is empty.")
        return None
    
    questions = [item['question'] for item in knowledge_base]
    answers = [item['answer'] for item in knowledge_base]
    
    # Ensure we have valid training data
    if not questions or not answers:
        print("Not enough questions or answers to train the model.")
        return None
    
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(questions, answers)
    print(f"Model trained with {len(questions)} entries.")
    return model
