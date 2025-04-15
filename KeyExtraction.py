import os
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import pytextrank
import spacy
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk.stem import WordNetLemmatizer
import re  # Import regular expression library
import networkx as nx
import matplotlib.pyplot as plt
import string

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
            keywords = [lemmatizer.lemmatize(re.sub(r',\s*\d+$', '', line.strip().lower())) for line in file]
        return set(keywords)
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return None

reference_keywords = load_keywords_from_file(keywords_file)

if not reference_keywords:
    print("No keywords loaded from the file.")
    exit()

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_keywords_pytextrank(text, num_keywords=200):
    doc = nlp(text)
    keywords = [
        lemmatizer.lemmatize(phrase.text.strip().lower())
        for phrase in doc._.phrases
        if not any(char in string.punctuation for char in phrase.text)
    ]
    return list(set(keywords[:num_keywords]))

def evaluate_keywords(reference, predicted):
    true_positives = 0
    for kw in predicted:
        if kw in reference:
            true_positives += 1

    precision = 0
    if len(predicted) > 0:
        precision = true_positives / len(predicted)

    recall = 0
    if len(reference) > 0:
        recall = true_positives / len(reference)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def build_concept_map(keywords, text):
    """Builds a concept map from the extracted keywords and text."""
    G = nx.Graph()
    unique_keywords = list(set(keywords)) # Use unique extracted keywords
    G.add_nodes_from(unique_keywords)

    # Simple co-occurrence based relationship detection (adjust window_size as needed)
    window_size = 5
    words = re.findall(r'\b\w+\b', text.lower()) # Tokenize the text

    for i in range(len(words) - window_size + 1):
        window = words[i : i + window_size]
        for j, word1 in enumerate(window):
            if word1 in unique_keywords:
                for k, word2 in enumerate(window):
                    if j < k and word2 in unique_keywords:
                        if G.has_edge(word1, word2):
                            G[word1][word2]['weight'] = G[word1][word2].get('weight', 0) + 1
                        else:
                            G.add_edge(word1, word2, weight=1)

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout algorithm
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', width=[d['weight'] for (u, v, d) in G.edges(data=True)])
    plt.title("Concept Map")
    plt.show()

print("\nProcessing Training Data...")
for filename in os.listdir(train_dir):
    train_file_path = os.path.join(train_dir, filename)
    if os.path.isfile(train_file_path):
        train_text = load_text_from_file(train_file_path)
        if train_text:
            extracted_keywords_train = extract_keywords_pytextrank(train_text)
            num_extracted_train = len(extracted_keywords_train)
            print(f"Processed training file: {filename}, Extracted Keywords: {num_extracted_train}")

print("\nEvaluating on Test Data...")
all_extracted_keywords = {} # Store extracted keywords for each test file.
all_test_texts = {} # Store test texts with filenames as keys

all_precisions = {}
all_recalls = {}
all_f1s = {}

for filename in os.listdir(test_dir):
    test_file_path = os.path.join(test_dir, filename)
    if os.path.isfile(test_file_path):
        test_text = load_text_from_file(test_file_path)
        if test_text:
            extracted_keywords = extract_keywords_pytextrank(test_text)
            num_extracted = len(extracted_keywords)

            all_extracted_keywords[filename] = extracted_keywords
            all_test_texts[filename] = test_text

            precision, recall, f1 = evaluate_keywords(reference_keywords, extracted_keywords)
            all_precisions[filename] = precision
            all_recalls[filename] = recall
            all_f1s[filename] = f1

            print(f"File: {filename}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Extracted Keywords: {num_extracted}")
            print(f"   First 10 Extracted: {extracted_keywords[:10]}")
            # print(f"   Reference: {reference_keywords}")

if all_f1s:
    avg_precision = sum(all_precisions.values()) / len(all_precisions)
    avg_recall = sum(all_recalls.values()) / len(all_recalls)
    avg_f1 = sum(all_f1s.values()) / len(all_f1s)

    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-score: {avg_f1:.4f}")
else:
    print("No test files processed.")

# Build and display the concept map after processing all test files
if all_extracted_keywords and all_test_texts:
    combined_extracted_keywords = []
    combined_text = ""
    for filename in all_test_texts:
        combined_extracted_keywords.extend(all_extracted_keywords[filename])
        combined_text += all_test_texts[filename] + " "
    build_concept_map(combined_extracted_keywords, combined_text)