import streamlit as st
import random

# Questions to ask the user
questions = [
    "What is your name?",
    "How old are you?",
    "What is your native language?",
    "How long have you been learning French?",
    "How often do you practice French?",
    "What is your highest level of education?",
    "Why do you want to learn French?",
    "What are your hobbies?",
    "Do you have any previous experience with learning languages?",
    "What kind of content do you prefer for learning?"
]

# Function to get 5 random questions
def get_random_questions(questions, num_questions=5):
    return random.sample(questions, num_questions)

# Streamlit app title
st.title("French Learning Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "questions" not in st.session_state:
    st.session_state.questions = get_random_questions(questions)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if we are done with questions
if st.session_state.current_question < len(st.session_state.questions):
    # Ask the next question
    question = st.session_state.questions[st.session_state.current_question]
    if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["content"] != question:
        st.session_state.messages.append({"role": "assistant", "content": question})
    
    with st.chat_message("assistant"):
        st.markdown(question)

    # Accept user input
    prompt = st.chat_input("Your answer:")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.answers.append(prompt)
        st.session_state.current_question += 1

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
else:
    st.write("Thank you for answering all the questions!")
    st.write("Here are your answers:")
    for i, answer in enumerate(st.session_state.answers):
        st.write(f"Q{i+1}: {st.session_state.questions[i]}")
        st.write(f"A{i+1}: {answer}")
    
    # Confirm answers
    if st.button("Confirm"):
        st.write("Your answers have been confirmed.")
        # Proceed with further steps, e.g., save answers, start the next phase of the app
    elif st.button("Start Over"):
        st.session_state.current_question = 0
        st.session_state.answers = []
        st.session_state.messages = []
        st.session_state.questions = get_random_questions(questions)
