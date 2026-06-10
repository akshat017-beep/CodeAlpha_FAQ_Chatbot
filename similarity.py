import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot.preprocess import preprocess_text

with open('faq_data.json', 'r') as file:
    faqs = json.load(file)

questions = [faq['question'] for faq in faqs]

processed_questions = [preprocess_text(q) for q in questions]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

def get_best_answer(user_question):

    processed_input = preprocess_text(user_question)

    input_vector = vectorizer.transform([processed_input])

    similarities = cosine_similarity(
        input_vector,
        question_vectors
    )

    best_match_index = similarities.argmax()

    confidence = similarities[0][best_match_index]

    if confidence < 0.2:
        return "Sorry, I couldn't understand your question."

    return faqs[best_match_index]['answer']