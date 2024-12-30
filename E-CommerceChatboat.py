import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Knowledge base with e-commerce related questions and answers
knowledge_base = {
    "general": {
        "questions": [
            "What are your store hours?",
            "Do you have a return policy?",
            "How can I contact customer service?"
        ],
        "answers": [
            "Our store operates 24/7 online. Physical store hours are 9 AM to 9 PM, Monday to Saturday.",
            "Yes, we have a 30-day return policy. Please ensure the product is unused and in its original packaging.",
            "You can contact our customer service at support@example.com or call 123-456-7890."
        ]
    },
    "orders": {
        "questions": [
            "Where is my order?",
            "How do I track my shipment?",
            "Can I cancel my order?"
        ],
        "answers": [
            "To check your order status, log in to your account and go to 'My Orders'.",
            "You can track your shipment using the tracking ID sent to your email after dispatch.",
            "Yes, you can cancel your order before it is shipped. Go to 'My Orders' and click 'Cancel Order'."
        ]
    },
    "products": {
        "questions": [
            "Do you have discounts on electronics?",
            "What is the warranty on your products?",
            "Do you offer international shipping?"
        ],
        "answers": [
            "Yes, we currently have discounts on selected electronics. Visit our 'Deals' section for more details.",
            "Most products come with a one-year manufacturer warranty. Please check the product page for specific details.",
            "Yes, we offer international shipping to select countries. Shipping fees and times vary based on location."
        ]
    }
}

# Preprocessing function
def preprocess_text(text):
    """Clean and preprocess user input."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Find the best answer based on cosine similarity
def get_answer(user_input):
    """Match user input to knowledge base and return an answer."""
    user_input = preprocess_text(user_input)
    all_questions = []
    all_answers = []

    for category, data in knowledge_base.items():
        all_questions.extend(data['questions'])
        all_answers.extend(data['answers'])

    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(all_questions)
    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    best_match = similarities.argmax()

    if similarities[best_match] > 0.2:  # Threshold for matching
        return all_answers[best_match]
    else:
        return "I'm sorry, I couldn't find an answer to that question. Please contact support for further assistance."

# Streamlit app
def main():
    st.title("E-Commerce Chatbot")
    st.write("Welcome to our e-commerce assistant! Ask me anything about our products, orders, or policies.")

    # User input
    user_input = st.text_input("Type your question below:")

    if user_input:
        answer = get_answer(user_input)
        st.write(f"**Chatbot:** {answer}")

    st.write("\n\nType your query and press Enter. Examples: 'What is your return policy?' or 'How do I track my order?'")

# Run the app
if __name__ == "__main__":
    main()
