from eatc_app import input_validation, process_data
import pandas as pd


''' MOCK OBJECTS '''
# Form a user_input dictionary
mock_valid_input_obj = {
    "Username": "jameslee",
    "Tweet": "Don't worry, happiness is close by. bro... where?",
    "Retweet_count": 0,
    "Mention_count": 0,
    "Follower_count": 0,
    "Verified": False,
    "Hashtags": "life, funny"
}

''' USER INPUT VALIDATION TEST CASES'''
# All valid input - success
def test_validInput_success():
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

# Username - empty username - error
def test_emptyUsername_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Username"] = ""
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Please ensure there are no empty values."

# Username - username with space - error
def test_usernameWithSpacing_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Username"] = "james lee"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username can only contain letters, numbers, and underscores (_)."

# Username - username with spcial chara - error
def test_usernameWithSpecialChara_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Username"] = "james@lee"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username can only contain letters, numbers, and underscores (_)."

# Username - username that is 'Admin' or 'X' - error
def test_AdminOrXusername_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Username"] = "Admin"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username cannot be 'Admin' or 'X'."

    invalid_obj["Username"] = "X"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username cannot be 'Admin' or 'X'."

# Username - username shorten then 4 or longer than 15 characters - error
def test_usernameOutsideLengthRange_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Username"] = "Hii"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username must be between 4 and 15 characters."

    invalid_obj["Username"] = "1234567890123456"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Username must be between 4 and 15 characters."

# Tweet - empty tweet - error
def test_emptyTweet_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Tweet"] = ""
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Please ensure there are no empty values."

# Tweet - tweet that is too long - error
def test_tooLongTweet_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Tweet"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Tweet must be equal or below 280 characters long."

# Retweet_count - zero or positive Retweet_count - success
def test_validRetweetCount_success():
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

    mock_valid_input_obj["Retweet_count"] = 10
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

# Retweet_count - negative number - error
def test_negativeRetweetCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Retweet_count"] = -1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Reweet count must not be a negative number."

# Retweet_count - negative 0 number - success
def test_negativezeroRetweetCount_error():
    valid_obj = mock_valid_input_obj.copy()
    valid_obj["Retweet_count"] = -0
    result, err_msg = input_validation(valid_obj)
    assert result == True, err_msg == ""

# Retweet_count - float number - error
def test_floatRetweetCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Retweet_count"] = 0.1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Reweet count must be a whole number."

# Mention_count - zero or positive Mention_count - success
def test_validMentionCount_success():
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

    mock_valid_input_obj["Mention_count"] = 10
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

# Mention_count - negative number - error
def test_negativeMentionCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Mention_count"] = -1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Mention count must not be a negative number."

# Mention_count - float number - error
def test_floatMentionCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Mention_count"] = 0.1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Mention count must be a whole number."

# Follower_count - zero or positive Follower_count - success
def test_validFollowerCount_success():
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

    mock_valid_input_obj["Follower_count"] = 10
    result, err_msg = input_validation(mock_valid_input_obj)
    assert result == True, err_msg == ""

# Follower_count - negative number - error
def test_negativeFollowerCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Follower_count"] = -1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Follower count must not be a negative number."

# Follower_count - float number - error
def test_floatRetweetCount_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Follower_count"] = 0.1
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Follower count must be a whole number."

# Hashtag - hashtags not between 2-63 characters long - error
def test_hashtagLengthError_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Hashtags"] = "s"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Hashtags must be between 2 and 63 characters excluding the '#' symbol."

    invalid_obj["Hashtags"] = "ziabnowaqojktfgylmdfxgswtxlsbtfonjlulnwtlnbivgmctlkdvufzosdbixfz"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Hashtags must be between 2 and 63 characters excluding the '#' symbol."

# Hashtag - empty hashtag - error
def test_emptyHashtag_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Hashtags"] = ""
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Please ensure there are no empty values."

    invalid_obj["Hashtags"] = "#"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Please enter in a valid hashtag value."

# Hashtag - hashtag with special characters - error
def test_hashtagWithSpecialChara_error():
    invalid_obj = mock_valid_input_obj.copy()
    invalid_obj["Hashtags"] = "@-;12"
    result, err_msg = input_validation(invalid_obj)
    assert result == False, err_msg == "Hashtags can only contain letters, numbers, and underscores (_)."

''' PROCESS_DATA() TEST CASES '''
# Check if the resulting dataframe values are correct
def test_process_data():
    # Convert user_input dictionary into a dataframe
    mock_valid_input_df = pd.DataFrame([mock_valid_input_obj])

    processed_df = process_data(mock_valid_input_df)
    tweet_embeded_columns = processed_df.columns[processed_df.columns.str.contains("tweet_embed")]
    tweet_embeded_df = processed_df[tweet_embeded_columns]
    structured_df = processed_df.drop(tweet_embeded_columns.tolist(), axis=1)

    # Check if the number of columns are generated properly
    assert processed_df.shape == (1, 401)
    assert tweet_embeded_df.shape == (1, 384)
    assert structured_df.shape == (1, 17)
    
    # Check if values are correct
    assert processed_df['Retweet_count'][0] == 10
    assert processed_df['Mention_count'][0] == 10
    assert processed_df['Follower_count'][0] == 10
    assert processed_df['Verified'][0] == 0

    # Check if Derived values are correct
    assert processed_df['Username_length'][0] == 8
    assert processed_df['Username_digitRatio'][0] == 0
    assert processed_df['Username_firstLastNameNumberPattern'][0] == 0
    assert processed_df['Tweet_length'][0] == 49
    assert processed_df['Tweet_wordCount'][0] == 8
    assert processed_df['Tweet_upperCaseRatio'][0] == 0.02040816326530612
    assert processed_df['Tweet_hasJoinMsg'][0] == 0
    assert processed_df['Tweet_hasPromotion'][0] == 0
    assert processed_df['Follower_per_Mention'][0] == 0.9090909090909091
    assert processed_df['Follower_per_Retweet'][0] == 0.9090909090909091
    assert processed_df['hashtag_count'][0] == 2
    assert processed_df['hashtag_repeationRatio'][0] == 0
    assert processed_df['Has_Hashtag'][0] == 1

# Check if the resulting dataframe values are correct
def test_process_data_specialCases():
    # Convert user_input dictionary into a dataframe
    mock_valid_input_df = pd.DataFrame([mock_valid_input_obj])
    mock_valid_input_df.loc[0, 'Hashtags'] = "none"
    mock_valid_input_df.loc[0, 'Tweet'] = "Please subscribe to win a 1 for 1 free deal! Come check it out!"

    processed_df = process_data(mock_valid_input_df)

    # Check if Changed values are correct
    assert processed_df['Tweet_length'][0] == 63
    assert processed_df['Tweet_wordCount'][0] == 14
    assert processed_df['Tweet_upperCaseRatio'][0] == 0.031746031746031744
    assert processed_df['Tweet_hasJoinMsg'][0] == 1
    assert processed_df['Tweet_hasPromotion'][0] == 1

    assert processed_df['hashtag_count'][0] == 0
    assert processed_df['hashtag_repeationRatio'][0] == 0
    assert processed_df['Has_Hashtag'][0] == 0