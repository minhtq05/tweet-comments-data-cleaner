import re


def remove_internet_contents(text):
    # text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub("www.[A-Za-z0-9-?[-`{-~]", "", text)  # remove urls
    text = re.sub("@[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove usernames
    text = re.sub("#[A-Za-z0-9!-?[-`{-~]+", "", text)  # remove hashtags
    return text
