# Initialize st.session_state.new_session as True for new users (excluding vishakha)
if 'new_session' not in st.session_state and st.session_state.user_name != "vishakha":
    st.session_state.new_session = True

# Check if the user's name is "vishakha"
if st.session_state.user_name == "vishakha":
    is_admin = True
    st.session_state.user_role = "admin"
    st.session_state.user_name = user_name
    st.session_state.new_session = False  # Prevent clearing chat history
    st.session_state.sessions = load_previous_sessions()
else:
    # Admin-specific session handling
    if st.session_state.user_name_input == "admin" and st.session_state.user_name != "vishakha":
        is_admin = True
        st.session_state.user_role = "admin"
        st.session_state.user_name = user_name
        st.session_state.new_session = False  # Prevent clearing chat history
        st.session_state.sessions = load_previous_sessions()
