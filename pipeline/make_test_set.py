import random

import nltk
from config import Config

f = open(Config.RAW_DATA)
corpus = f.read()

# Split into sentences
sent_list = nltk.tokenize.sent_tokenize(corpus)
# Remove any sentences that are suspiciously short - say <= 20 characters
clean_list = [s for s in sent_list if len(s) > 20]

# Randomly select 1000 for testing
random.seed(Config.RANDOM_SEED)
keep = random.sample(clean_list, 500)

# Write this subset to file
with open(Config.TEST_SET, "w") as f:
    for item in keep:
        # Remove any newlines in the body of the text to avoid confusion
        f.write("%s\t" % item.strip())
