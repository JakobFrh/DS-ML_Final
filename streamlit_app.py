import streamlit as st
import random
import time

# Questions to ask the user
questions = [
    "What is your name?",
    "How old are you?",
    "What is your native language?",
    "How long have you been learning English?",
    "How often do you practice English?",
    "What is your highest level of education?"
]

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Thank you for your response.",
            "Got it, thank you!",
            "Thanks for the information."
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Language Level Prediction Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "answers" not in st.session_state:
    st.session_state.answers = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if we are done with questions
if st.session_state.current_question < len(questions):
    # Ask the next question
    question = questions[st.session_state.current_question]
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

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = ''.join(response_generator())
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Thank you for answering all the questions!")
    st.write("Your answers have been saved.")
    st.write(st.session_state.answers)

    # Option to reset and start over
    if st.button("Start Over"):
        st.session_state.current_question = 0
        st.session_state.answers = []
        st.session_state.messages = []
