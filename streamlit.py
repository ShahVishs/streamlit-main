import os
from airtable import Airtable
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pytz import timezone
from datetime import datetime
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
import json

# CSS to hide the Share button
hide_share_button_style = """
    <style>
    .st-emotion-cache-zq5wmm.ezrtsby0 .stActionButton:nth-child(1) {
        display: none !important;
    }
    </style>
"""

# CSS to hide the Star and GitHub elements
hide_star_and_github_style = """
    <style>
    .st-emotion-cache-1lb4qcp.e3g6aar0,
    .st-emotion-cache-30do4w.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

# CSS to hide the MainMenu
hide_mainmenu_style = """
    <style>
    #MainMenu {
        display: none !important;
    }
    </style>
"""

# CSS to hide the "Fork this app" button
hide_fork_app_button_style = """
    <style>
    .st-emotion-cache-alurl0.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

# Apply the CSS styles
st.markdown(hide_share_button_style, unsafe_allow_html=True)
st.markdown(hide_star_and_github_style, unsafe_allow_html=True)
st.markdown(hide_mainmenu_style, unsafe_allow_html=True)
st.markdown(hide_fork_app_button_style, unsafe_allow_html=True)

# Custom CSS style to set the avatar as your logo
custom_avatar_style = """
<style>
    .custom-avatar {
        background-image: url("https://github.com/ShahVishs/streamlit-main/blob/main/logo.png"); /* Replace with the path to your logo image */
        background-size: cover;
        width: 40px; /* Adjust the width and height as needed */
        height: 40px;
        border-radius: 50%; /* To make it circular */
        margin-right: 10px; /* Adjust the margin as needed */
    }
</style>
"""

# Apply the custom CSS style
st.markdown(custom_avatar_style, unsafe_allow_html=True)
# # Display the image
# st.image("Twitter.jpg", caption="Twitter.jpg", use_column_width=True, output_format="JPEG", key="image", container_class="image-container")
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("Twitter.jpg")
# datetime.datetime.now()
datetime.now()
# Get the current date in "%m/%d/%y" format
# current_date = datetime.date.today().strftime("%m/%d/%y")
current_date = datetime.today().strftime("%m/%d/%y")
# Get the day of the week (0: Monday, 1: Tuesday, ..., 6: Sunday)
# day_of_week = datetime.date.today().weekday()
day_of_week = datetime.today().weekday()
# Convert the day of the week to a string representation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

# print("Current date:", current_date)
# print("Current day:", current_day)
todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm"
    "Phone: (555) 123-4567"
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()
current_date = datetime.today().strftime("%m/%d/%y")
day_of_week = datetime.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Define roles (e.g., 'admin' and 'user')
ROLES = ['admin', 'user']

# Initialize user role in session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# Initialize session-specific chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the sessions variable in session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
    
# Initialize a dictionary to cache responses for repeated questions
question_cache = {}

# Function to save chat session data
def save_chat_session(session_data, session_id):
    session_directory = "chat_sessions"
    session_filename = f"{session_directory}/chat_session_{session_id}.json"

    if not os.path.exists(session_directory):
        os.makedirs(session_directory)

    session_dict = {
        'user_name': session_data['user_name'],
        'chat_history': session_data['chat_history']
    }

    try:
        with open(session_filename, "w") as session_file:
            json.dump(session_dict, session_file)
    except Exception as e:
        st.error(f"An error occurred while saving the chat session: {e}")

# Load previous sessions
def load_previous_sessions():
    previous_sessions = {}

    if not os.path.exists("chat_sessions"):
        os.makedirs("chat_sessions")

    session_files = os.listdir("chat_sessions")

    for session_file in session_files:
        session_filename = os.path.join("chat_sessions", session_file)
        session_id = session_file.split("_")[-1].split(".json")[0]

        with open(session_filename, "r") as session_file:
            session_data = json.load(session_file)
            previous_sessions[session_id] = session_data

    return previous_sessions

# Initialize st.session_state.past as an empty list if it doesn't exist
if 'past' not in st.session_state:
    st.session_state.past = []

# Initialize st.session_state.new_session as True
if 'new_session' not in st.session_state:
    st.session_state.new_session = True

# Initialize user name input
if 'user_name_input' not in st.session_state:
    st.session_state.user_name_input = None

# Refresh Session Button
if st.button("Refresh Session"):
    # Save the current session and start a new one
    current_session = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }

    # Generate a unique session_id based on the timestamp
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    save_chat_session(current_session, session_id)

    # Clear session state variables to start a new session
    st.session_state.chat_history = []
    st.session_state.user_name = None
    st.session_state.user_name_input = None
    st.session_state.new_session = True
    st.session_state.refreshing_session = False  # Reset refreshing_session to False

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Check if it's a new session
if st.session_state.new_session:
    # Load previous chat history for the current user if available
    user_name = st.session_state.user_name
    if user_name:
        previous_sessions = load_previous_sessions()
        if user_name in previous_sessions:
            st.session_state.chat_history = previous_sessions[user_name]['chat_history']
    st.session_state.new_session = False

# Display a list of past sessions in the sidebar along with a delete button
st.sidebar.header("Chat Sessions")

# Check if the user is the admin (vishakha) or not
is_admin = st.session_state.user_name == "vishakha"

# Create a dictionary to store sessions for each user
user_sessions = {}

for session_id, session_data in st.session_state.sessions.items():
    user_name = session_data['user_name']
    chat_history = session_data['chat_history']

    if user_name not in user_sessions:
        user_sessions[user_name] = []

    user_sessions[user_name].append({
        'session_id': session_id,
        'chat_history': chat_history
    })

# If the user is an admin (vishakha), show all sessions for all users
if st.session_state.user_name == "vishakha":
    for user_name, sessions in user_sessions.items():
        for session in sessions:
            formatted_session_name = f"{user_name} - {session['session_id']}"

            button_key = f"session_button_{session['session_id']}"
            if st.sidebar.button(formatted_session_name, key=button_key):
                st.session_state.chat_history = session['chat_history'].copy()
else:
    # If the user is not an admin, show only their own session
    user_name = st.session_state.user_name
    if user_name:
        if user_name in user_sessions:
            for session in user_sessions[user_name]:
                formatted_session_name = f"{user_name} - {session['session_id']}"

                if st.sidebar.button(formatted_session_name):
                    st.session_state.chat_history = session['chat_history'].copy()


file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 8})#check without similarity search and k=8

# Create the first tool
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_dealership_inventory",
     "Searches and returns documents regarding the car inventory and Input should be a single string strictly."
)

# Create the third tool
tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# Append all tools to the tools list
tools = [tool1, tool3]

# airtable
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appAVFD4iKFkBm49q"  
AIRTABLE_TABLE_NAME = "Question_Answer_Data" 
# Streamlit UI setup
st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")

if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Check if the user's name is "vishakha"
# Check if the user's name is "vishakha"
if st.session_state.user_name == "vishakha":
    is_admin = True
    st.session_state.user_role = "admin"
    st.session_state.user_name = user_name
    st.session_state.new_session = False  # Prevent clearing chat history
    st.session_state.sessions = load_previous_sessions()
else:
    # Initialize st.session_state.new_session as True for new users (excluding vishakha)
    if 'new_session' not in st.session_state and st.session_state.user_name != "vishakha":
        st.session_state.new_session = True
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature = 0)
    langchain.debug=True
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    template=(
    """You're the Business Development Manager at our car dealership./
    When responding to inquiries, please adhere to the following guidelines:
    Car Inventory Questions: If the customer's inquiry lacks specific details such as their preferred/
    make, model, new or used car, and trade-in, kindly engage by asking for these specifics./
    Specific Car Details: When addressing questions about a particular car, limit the information provided/
    to make, year, model, and trim. For example, if asked about 
    'Do you have Jeep Cherokee Limited 4x4'
    Best answer should be 'Yes we have,
    Jeep Cherokee Limited 4x4:
    Year: 2022
    Model :
    Make :
    Trim:
    scheduling Appointments: If the customer's inquiry lacks specific details such as their preferred/
    day, date or time kindly engage by asking for these specifics. {details} Use these details that is todays date and day /
    to find the appointment date from the users input and check for appointment availabity for that specific date and time. 
    If the appointment schedule is not available provide this 
    link: www.dummy_calenderlink.com to schedule appointment by the user himself. 
    If appointment schedules are not available, you should send this link: www.dummy_calendarlink.com to the 
    costumer to schedule an appointment on your own.

    Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or/
    receive product briefings from our team. After providing essential information on the car's make, model,/
    color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us/
    for a comprehensive product overview by our experts.

    Please maintain a courteous and respectful tone in your American English responses./
    If you're unsure of an answer, respond with 'I am sorry.'/
    Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences."
    Feel free to use any tools available to look up for relevant information.
    Answer the question not more than two sentence.""")

    details = "Today's current date is " + todays_date + " and today's week day is " + day_of_the_week + "."

    input_template = template.format(details=details)

    system_message = SystemMessage(
        content=input_template)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    if 'agent_executor' not in st.session_state:
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
        st.session_state.agent_executor = agent_executor
    else:
        agent_executor = st.session_state.agent_executor
    response_container = st.container()
    airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)
    # Function to save chat data to Airtable
   
    def save_chat_to_airtable(user_name, user_input, output):
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            airtable.insert(
                {
                    "username": user_name,
                    "question": user_input,
                    "answer": output,
                    "timestamp": timestamp,
                }
            )
        except Exception as e:
            st.error(f"An error occurred while saving data to Airtable: {e}")

    # Function for conversational chat
    # @st.cache_data
    # def conversational_chat(user_input):
    #     result = agent_executor({"input": user_input})
    #     st.session_state.chat_history.append((user_input, result["output"]))
    #     return result["output"]
    # Function for conversational chat
    # @st.cache_data
    # def conversational_chat(user_input):
    #     # Check if the user has asked this question before
    #     previous_answer = get_previous_answer_from_airtable(user_input)
        
    #     if previous_answer:
    #         return previous_answer
        
    #     result = agent_executor({"input": user_input})
    #     st.session_state.chat_history.append((user_input, result["output"]))
    #     return result["output"]
    # Function to remember the conversation for a session and cache responses
    # def conversational_chat(user_input):
    #     # Check if the user's input is in the cache
    #     if user_input in question_cache:
    #         # If yes, retrieve the cached response
    #         output = question_cache[user_input]
    #     else:
    #         # If no, generate a new response and append to chat history
    #         # You should include your code for generating AI responses here
    #         # For demonstration, let's assume we generate a simple response
    #         output = "AI Response: " + user_input
    #         st.session_state.chat_history.append((user_input, output))
    #         # Cache the response for future use
    #         question_cache[user_input] = output
    
    #     return output
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []   
    # Function for conversational chat
    # Function for conversational chat
    def conversational_chat(user_input):
        # Check if the user's question matches any previous questions in chat history
        for query, answer in reversed(st.session_state.chat_history):
            if query.lower() == user_input.lower():  # Case-insensitive comparison
                # If a match is found, return both the question and its answer
                  return answer
        
        # If not found in history, continue the conversation with the AI agent
        result = agent_executor({"input": user_input})
        response = result["output"]
        
        # Append the current question and its response to chat history
        # st.session_state.chat_history.append((user_input, response))
        
        return response
    # def get_previous_answer_from_airtable(user_input):
    #     try:
    #         # Query Airtable to check if there's a previous answer for this question
    #         records = airtable.search('question', user_input)
            
    #         if records:
    #             # Assuming you're only interested in the first matching record
    #             previous_answer = records[0]['fields']['answer']
    #             return previous_answer
    #         else:
    #             # Handle the case when no records are found
    #             return None
    #     except Exception as e:
    #         # Log the error for debugging
    #         st.error(f"An error occurred while querying Airtable: {e}")
    #         return None  # Return None or an appropriate error message
            
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
        if user_name == "vishakha":
            # Load chat history for "vishakha" without asking for a query
            is_admin = True
            st.session_state.user_role = "admin"
            st.session_state.user_name = user_name
            st.session_state.new_session = False  # Prevent clearing chat history
            st.session_state.sessions = load_previous_sessions()
   
    user_input = ""
    output = ""
    
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
        if user_name == "vishakha":
            # Load chat history for "vishakha" without asking for a query
            is_admin = True
            st.session_state.user_role = "admin"
            st.session_state.user_name = user_name
            st.session_state.new_session = False  # Prevent clearing chat history
            st.session_state.sessions = load_previous_sessions()
    
    with st.form(key='my_form', clear_on_submit=True):
        if st.session_state.user_name != "vishakha":
            user_input = st.text_input("Query:", placeholder="Type your question here :)", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        response = conversational_chat(user_input)
        # Append the user's question and its answer to the chat history
        st.session_state.chat_history.append((user_input, response))
    # GitHub repository URL for the image
    # image_url = 'https://raw.githubusercontent.com/ShahVishs/streamlit-main/blob/main/icon-1024.png'
    # Inside your Streamlit app:

    with response_container:
        for i, (query, answer) in enumerate(st.session_state.chat_history):
            user_name = st.session_state.user_name
            # message(query, is_user=True, key=f"{i}_user", avatar_style="icons", seed=6)
            message(query, is_user=True, key=f"{i}_user", avatar_style="avataaars", seed=6)
            # Display the user message on the right
            # col1, col2 = st.columns([1, 8])  # Adjust the ratio as needed
            # with col1:
            #     st.image("icons8-user-96.png", width=50)
            # with col2:
            #     st.markdown(
            #         f'<div style="background-color: #DCF8C6; border-radius: 10px; padding: 10px; width: 70%;'  # Adjusted width here
            #         f' border-top-right-radius: 0; border-bottom-right-radius: 0;'
            #         f' border-top-left-radius: 10px; border-bottom-left-radius: 10px; box-shadow: 2px 2px 5px #888888; margin-bottom: 10px;">'
            #         f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{query}</span>'
            #         f'</div>',
            #         unsafe_allow_html=True
            #     )
    
            # # Display the response on the left
            # col3, col4 = st.columns([1, 8])  # Adjust the ratio as needed
            # with col3:
            #     st.image("icon-1024.png", width=50)
            # with col4:
            #     st.markdown(
            #         f'<div style="background-color: #F5F5F5; border-radius: 10px; padding: 10px; width: 70%;'  # Adjusted width here
            #         f' border-top-right-radius: 0; border-bottom-right-radius: 0;'
            #         f' border-top-left-radius: 10px; border-bottom-left-radius: 10px; box-shadow: 2px 2px 5px #888888; margin-bottom: 10px;">'
            #         f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
            #         f'</div>',
            #         unsafe_allow_html=True
            #     )
    
            # # Add some spacing between question and answer
            # st.write("")
            # Display the logo image
            # st.image("icon-1024.png", width=40)
        
            # Display the answer with the desired avatar style
            # message(answer, key=f"{i}_answer", avatar_style="initials", seed="AI",)
            # Display the logo image for the user's query
            # st.image("icon-1024.png", width=40)
    
            # # Display the answer without avatars
            col1, col2 = st.columns([0.7, 10])  # Adjust the ratio as needed
            with col1:
                st.image("icon-1024.png", width=50)
            with col2:
                st.markdown(
                f'<div style="background-color: #F5F5F5; border-radius: 10px; padding: 10px; width: 50%;'
                f' border-top-right-radius: 10px; border-bottom-right-radius: 10px;'
                f' border-top-left-radius: 0; border-bottom-left-radius: 0; box-shadow: 2px 2px 5px #888888;">'
                f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
                f'</div>',
                unsafe_allow_html=True
                )
                    
            # # Add a bit more space between the query and answer messages
            # st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            
            # # Wrap the answer in a div with background color and padding
            # st.markdown(
            #     f'<div style="background-color: #e0e0e0; border-radius: 5px; padding: 10px;">'
            #     f'{answer}'
            #     f'</div>',
            #     unsafe_allow_html=True
            # )
    if st.session_state.user_name and st.session_state.chat_history:
        try:
            save_chat_to_airtable(st.session_state.user_name, user_input, output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
