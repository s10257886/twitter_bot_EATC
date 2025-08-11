import streamlit as st
import pathlib
import pandas as pd
import re
import joblib


''' FUNCTIONS '''
# Process the input data to a fit the model
def process_data(df):

    # List of boolean columns
    bool_cols = ['Verified', 'Username_firstLastNameNumberPattern', 'Tweet_hasJoinMsg', 'Tweet_hasPromotion', 'Has_Hashtag']

    ### CREATION OF DERIVED FEATURES ###
    # Username - length, num of special chara, digit ratio, common bot patterns
    df["Username_length"] = df["Username"].apply(lambda x: len(str(x)))
    df["Username_digitRatio"] = df["Username"].apply(lambda x: (sum(c.isdigit() for c in x) / len(str(x))) if len(str(x))> 0 else 0)
    df["Username_firstLastNameNumberPattern"] = df["Username"].apply(lambda x: bool(re.search(r'^[a-z]+[a-z]+\d+$', x)))
    
    # Tweet 
    df["Tweet_length"] = df["Tweet"].apply(lambda x: len(str(x)))
    df["Tweet_wordCount"] = df["Tweet"].apply(lambda x: len(str(x).split()))
    df["Tweet_upperCaseRatio"] = df["Tweet"].apply(lambda x: (sum(c.isupper() for c in x) / len(str(x))) if len(str(x))> 0 else 0)
    
    df["Tweet_hasJoinMsg"] = df["Tweet"].apply(lambda x: bool(re.search(r'\b(follow|like|subscribe|join|click)\b', x.lower())))
    df["Tweet_hasPromotion"] = df["Tweet"].apply(lambda x: bool(re.search(r'\b(buy|sale|discount|offer|deal|free|win|prize)\b', x.lower())))
    
    # Follower Ratio features 
    df["Follower_per_Mention"] = df["Follower_count"] / (df["Mention_count"] + 1)
    df["Follower_per_Retweet"] = df["Follower_count"] / (df["Retweet_count"] + 1)

    # Hashtags
    def extract_hashtag_list(hashtags_text):
        # Parse hashtags - assuming they're separated by spaces, commas, or semicolons
        hashtags_str = str(hashtags_text).lower().strip()
        hashtags = re.split(r'[,;\s]+', hashtags_str)
        hashtags = [tag.strip() for tag in hashtags if tag.strip()]  # Remove empty tags
        return hashtags

    df["hashtag_count"] = df["Hashtags"].apply(lambda x: 0 if extract_hashtag_list(x) == ["none"] else len(extract_hashtag_list(x))) # CHANGED
    df["hashtag_repeationRatio"] = df["Hashtags"].apply(
        lambda x: 0 if extract_hashtag_list(x) == ["none"] else 
        1 - len(set(extract_hashtag_list(x))) / len(extract_hashtag_list(x))
    )
    df["Has_Hashtag"] = df["Hashtags"].apply(lambda x: False if extract_hashtag_list(x) == ["none"] else True)

    ### DROP UNNECESSARY COLUMNS ###
    df.drop(columns=["Username", "Hashtags"], inplace=True)
    
    ### ENCODING & EMBEDDING CATERGORICAL DATA ###
    # Encode Bool dataset
    encoding_guide = {True: 1, False: 0}
    for col in bool_cols:
        df[col] = df[col].map(encoding_guide)
    
    # Clean the 'Tweet' column
    def clean_tweet(tweet):
        text = str(tweet).lower() # change to lowercase
        text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text) # remove user handles or URL
        return text.strip() # remove leading and trailing characters from a string.
    df = df[df['Tweet'].notnull() & (df['Tweet'].str.strip() != '')] # remove short or empty tweets
    df['Tweet'] = df['Tweet'].apply(lambda x: clean_tweet(x))
    
    # Encode using SentenceTransformer
    from sentence_transformers import SentenceTransformer
    # Use a pretrained sentence embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tweet_embeddings = model.encode(df['Tweet'].tolist())  # shape: (n_samples, embedding_dim)

    ### COMBINE THE EMBEDDED TWEETS WITH OTHER FEATURES ###
    # Create DataFrame for tweet embeddings with meaningful column names
    tweet_embedding_df = pd.DataFrame(
        tweet_embeddings, 
        columns=[f'tweet_embed_{i}' for i in range(tweet_embeddings.shape[1])],
        index=df.index
    )
    
    # Get structured features (excluding Tweet and Bot Label)
    structured_features = df.drop(columns=['Tweet'])
    
    # Combine using concat to preserve all column names
    df = pd.concat([tweet_embedding_df, structured_features], axis=1)
    
    ### RETURN THE PROCESSED DATAFRAME ###
    return df

# Input validation
def input_validation(input_obj):

    # Check for empty values
    for col in input_obj.keys():
        # if input is empty or a "" string, return false
        if input_obj[col] == "":
            return False, "Please ensure there are no empty values."
        
    # Username - ensure no spacing, no special char, no 'Admin' or 'x', 4-15 characters long
    username = input_obj['Username']
    if not (4 <= len(username) <= 15):
        return False, "Username must be between 4 and 15 characters." + username
    if not re.match(r"^[A-Za-z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores (_)."
    if username.lower() in ["admin", "x"]:
        return False, "Username cannot be 'Admin' or 'X'."
    
    # Tweet - max 280 chara
    if len(input_obj['Tweet']) > 280:
        return False, "Tweet must be equal or below 280 characters long."
    
    # Number Inputs - no negative, no float
    number_cols = ['Retweet_count', 'Mention_count', 'Follower_count']
    for col in number_cols:
        if not isinstance(input_obj[col], int):
            return False, col.replace("_", " ") + " must be a whole number."
        if input_obj[col] < 0:
            return False, col.replace("_", " ") + " must not be a negative number."
    
    # Hashtag - each hashtag is 2-63 characters long, no special chara (except '_')
    split_hashtags = re.split(r"[ ,;]+", input_obj['Hashtags'])
    for tag in split_hashtags:
        # remove '#' symbol
        tag = tag.strip().lstrip("#")
        if not tag:
            return False, "Please enter in a valid hashtag value."
        # Only letters, numbers, underscore
        if not re.match(r"^[A-Za-z0-9_]+$", tag):
            return False, "Hashtags can only contain letters, numbers, and underscores (_)."
        if not (2 <= len(tag) <= 63):
            return False, "Hashtags must be between 2 and 63 characters excluding the '#' symbol."
        
    # Since all check pass, return true
    return True, ""

# Load CSS
def load_css(file_path):

    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

# Logging
def log(msg):
    with open("log.txt", "w") as file:
        file.write(msg)

''' MAIN CODE '''
### Load the external CSS ###
BASE_DIR = pathlib.Path(__file__).parent.resolve()

### Load the external CSS ###
css_path = BASE_DIR / "resources" / "styles" / "styles.css"
load_css(css_path)

### Import our trained model ###
model_path = BASE_DIR / "model.joblib"
loaded_model = joblib.load(model_path)

### Logo ###
sidebar_logo = BASE_DIR / "resources" / "images" / "hori_logo.png"
main_body_logo = BASE_DIR / "resources" / "images" / "hori_logo.png"
st.logo(str(sidebar_logo), icon_image=str(main_body_logo), size="large")  # Convert to str for Streamlit



#css_path = pathlib.Path("resources/styles/styles.css")
#load_css(css_path)

### Import our trained model ###
#loaded_model = joblib.load('model.joblib')

### Logo ###
#sidebar_logo = "resources/images/hori_logo.png"
#main_body_logo = "resources/images/hori_logo.png"
#st.logo(sidebar_logo, icon_image=main_body_logo, size="large") # side bar logo, icon_image= main body logo

### Website Header ###
#with st.container(key="header"):
 #   st.title("Detect Twitter Bots with AI Precision")
  #  st.write("Advanced machine learning algorithm to identify automated accounts and protect your social media presence")

### Website Body ###
body = st.container(key="body")
main_col1, main_col2, main_col3, main_col4 = body.columns([1,2,2,1])

# Column displaying the form
with main_col2:

    # Twitter Info form (with a class: "st-key-form")
    with st.form(key='form', width="stretch", height="stretch"):
        # Form Content
        with st.container(key="form_content"):
            # Form Headers
            st.markdown("### Analyze Twitter Account")
            st.write("Enter account details for bot detection analysis")

            # Error Message Container
            err_cont = st.empty()

            # Input Fields
            username = st.text_input('Username:', placeholder="Enter username", key="username_input", max_chars=15)
            verified = st.checkbox("Is a Verified Account", key="verified_input")
            tweet = st.text_area('Tweet Content:', placeholder="Enter tweet content (text only)", key="tweet_input", max_chars=280)

            retweet_col, mention_col, follower_col = st.columns(3)

            retweet_count = retweet_col.number_input("Repost Count:", min_value=0, help="number of repost of this tweet", key="retweet_input")
            mention_count = mention_col.number_input("Mention Count:", min_value=0, help="number of mentions in the tweet (eg @username)", key="mention_input")
            follower_count = follower_col.number_input("Follower Count:", min_value=0, help="number of follower the user has", key="follower_input")

            hashtags = st.text_input('Hashtags:', placeholder="Enter none or hashtags (comma, space, semicolon separated)", key="hashtag_input")
            
            submit_button = st.form_submit_button(label='Start Analysis', use_container_width=True)
            
            # When submit button is clicked
            if submit_button:
                # Form a user_input dictionary
                user_input = {
                    "Username": username,
                    "Tweet": tweet,
                    "Retweet_count": retweet_count,
                    "Mention_count": mention_count,
                    "Follower_count": follower_count,
                    "Verified": verified,
                    "Hashtags": hashtags
                }

                # Input Validation
                no_err, err_msg = input_validation(user_input)
                # If there is a input error, display error msg & set user_input to none in user session
                if not no_err:
                    err_cont.error(err_msg, icon="❗️")
                    st.session_state.user_input = None
                else:
                    # Ensure any error msg is cleared
                    err_cont.empty()
                    # Convert user_input dictionary into a dataframe
                    df_input = pd.DataFrame([user_input])
                    # Save user input in user session
                    st.session_state.user_input = df_input

# Column display the anaylsis results                
with main_col3:

    with st.container(key="result_box"):

        st.markdown("### Analysis Result")
        # if there is a user input in session and not none
        if 'user_input' in st.session_state and st.session_state.user_input is not None:

            # Clear previous result immediately
            st.session_state.prediction_result = None

            # Display the spinner (loading icon)
            with st.spinner("Analyzing...", show_time=True):
                
                # Process the user input
                df_input = st.session_state.user_input
                df_processed = process_data(df_input)

                # Predict using our trained model and save in user session
                pred = loaded_model.predict(df_processed)
                st.session_state.prediction_result = pred
                log("prediction:" + str(pred)) # logging

            # Display results
            with st.container(key="display_result"):

                # Show result only after prediction is ready
                if st.session_state.prediction_result is not None:

                    img_col1, img_col2, img_col3 = st.columns(3)

                    # if prediction = true (aka a bot)
                    if st.session_state.prediction_result == 1:
                        img_col2.image("resources/images/bot_img.png", caption="Twitter Bot")
                        st.markdown("#### This account has a high risk of being a Twitter Bot.")
                        st.error('This account shows characteristics of automated behavior. We ' \
                        'recommend exercising caution when engaging with this account.', icon="⚠️")
                    # if prediction = false (aka a human)
                    else: 
                        img_col2.image("resources/images/human_img.png", caption="Human User")
                        st.markdown("#### This account has a low risk of being a Twitter Bot.")
                        st.info('The result is not 100 percent accurate so exercising caution is still recommended.', icon="ℹ️")
        # else, if there isn't any user input
        else:
            img_col1, img_col2, img_col3 = st.columns([1,2,1])
            img_col2.image("resources/images/bored_bot.png", caption="I'm bored. Anything to analyze?")

### Website Footer ###
with st.container(key="footer"):
    st.divider()
    st.write("© 2025 BotSense. All rights reserved.")
                    
                    
                    

            





