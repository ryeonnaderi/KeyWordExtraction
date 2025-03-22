import os
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
import re
import networkx as nx
import matplotlib.pyplot as plt
from keybert import KeyBERT  # Import KeyBERT

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def get_directory_path(title):
    Tk().withdraw()
    dir_path = askdirectory(title=title)
    if not dir_path:
        print("No directory selected.")
    return dir_path

def get_file_path(title):
    Tk().withdraw()
    file_path = askopenfilename(title=title)
    if not file_path:
        print("No file selected.")
    return file_path

print("Select Training Data Directory:")
train_dir = get_directory_path("Select Training Data Directory")
print("Selected Training Directory:", train_dir)

print("\nSelect Test Data Directory:")
test_dir = get_directory_path("Select Test Data Directory")
print("Selected Test Directory:", test_dir)

print("\nSelect Keywords File:")
keywords_file = get_file_path("Select Keywords File")
print("Selected Keywords File:", os.path.basename(keywords_file))

if not (train_dir and test_dir and keywords_file):
    exit()

def load_text_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def load_keywords_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            keywords = [re.sub(r',\s*\d+$', '', line.strip().lower()) for line in file]
        return keywords
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return None

reference_keywords = load_keywords_from_file(keywords_file)

if not reference_keywords:
    print("No keywords loaded from the file.")
    exit()

nlp = spacy.load("en_core_web_sm") #Using Small spacy model.
kb = KeyBERT('distilbert-base-nli-mean-tokens') #Load KeyBERT model.

def extract_keywords_keybert(text, num_keywords=10):
    keywords = kb.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=num_keywords)
    return [keyword[0] for keyword in keywords]

def evaluate_keywords(reference, predicted):
    common = set(reference) & set(predicted)
    precision = len(common) / len(predicted) if predicted else 0
    recall = len(common) / len(reference) if reference else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def build_concept_map(keywords, text):
    """Builds a concept map from the extracted keywords and text."""
    G = nx.Graph()
    G.add_nodes_from(keywords)

    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords):
            if i < j:
                if keyword1 in text.lower() and keyword2 in text.lower():
                    G.add_edge(keyword1, keyword2)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
    plt.title("Concept Map")
    plt.show()

print("\nProcessing Training Data...")
for filename in os.listdir(train_dir):
    train_file_path = os.path.join(train_dir, filename)
    if os.path.isfile(train_file_path):
        train_text = load_text_from_file(train_file_path)
        if train_text:
            nlp(train_text) #Process training data with spacy.

print("\nEvaluating on Test Data...")
all_extracted_keywords = []
all_test_texts = []

all_precisions = []
all_recalls = []
all_f1s = []

for filename in os.listdir(test_dir):
    test_file_path = os.path.join(test_dir, filename)
    if os.path.isfile(test_file_path):
        test_text = load_text_from_file(test_file_path)
        if test_text:
            extracted_keywords = extract_keywords_keybert(test_text) #Using KeyBERT for keyword extraction.

            all_extracted_keywords.extend(extracted_keywords)
            all_test_texts.append(test_text)

            precision, recall, f1 = evaluate_keywords(reference_keywords, extracted_keywords)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            print(f"File: {filename}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
            
if all_f1s:
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print(f"\nAverage Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1-score: {avg_f1}")
else:
    print("No test files processed.")

if all_extracted_keywords and all_test_texts:
    combined_text = " ".join(all_test_texts)
    build_concept_map(all_extracted_keywords, combined_text)