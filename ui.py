import streamlit as st
import requests

st.set_page_config(page_title = "Rem AI", layout = "centered")

API_URL = "http://127.0.0.1:8001/chat"

st.title("Rem. The Memory Agent.")
st.write("Interract with Rem using this interface.")

thread_id_selection = st.text_area("Give a name for your session. ", height=68, placeholder="Session name here.. It helps to keep the context of a session without messing with other sessoins")

user_id_selection = st.text_area("Your name please", height = 68, placeholder= " Who is on the other end? This information will help me to have a personalized memory for you!")

user_input = st.text_area("Let's talk!",height = 68, placeholder = "Type your nessage here. I am all ears!")

if st.button("Submit"):
    if user_input.strip():
        try:
            #sends the payload to the FastAPI backend
            payload = {"input_message": user_input, "user_id": user_id_selection, 'thread_id': thread_id_selection}
            response = requests.post(API_URL, json=payload)

            # Display the response
            if response.status_code == 200:
                response_data = response.json()
                ai_responses = response_data.get("results", [])
                if ai_responses:
                    for response in ai_responses:
                        st.write(response)  # Display each AI response
                else:
                    st.error("No AI response found in the agent output.")


                    
            else:
                st.error(f"Request failed with status code {response.status_code}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Send Query'.")
