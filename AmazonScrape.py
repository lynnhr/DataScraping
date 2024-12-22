
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk

plt.style.use('ggplot')

# Attempt to read the CSV file with 'latin1' encoding
try:
    df = pd.read_csv('galaxys24ultra.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('galaxys24ultra.csv', encoding='ISO-8859-1')

example = df['Description'][50]

# Create the SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Description']
    author = row['Name']
    res[author] = sia.polarity_scores(text)

# Convert the results to a DataFrame
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Name'})
vaders = vaders.merge(df, how='left')

# Save the results to a CSV file
vaders.to_csv('vaders2.csv', index=False)
