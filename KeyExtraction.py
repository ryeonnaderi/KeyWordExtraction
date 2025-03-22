import os
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import pytextrank
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
import re  # Import regular expression library
import networkx as nx
import matplotlib.pyplot as plt

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
            keywords = [re.sub(r',\s*\d+$', '', line.strip().lower()) for line in file] # Remove numbers and lowercase
        return keywords
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return None

reference_keywords = load_keywords_from_file(keywords_file)

if not reference_keywords:
    print("No keywords loaded from the file.")
    exit()

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_keywords_pytextrank(text, num_keywords=10):
    doc = nlp(text)
    keywords = [(phrase.text.strip().lower(), phrase.rank) for phrase in doc._.phrases] # Lowercase and strip
    return keywords[:num_keywords]

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

    # Simple co-occurrence based relationship detection
    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords):
            if i < j:
                if keyword1 in text.lower() and keyword2 in text.lower():
                    G.add_edge(keyword1, keyword2)

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout algorithm
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
    plt.title("Concept Map")
    plt.show()

print("\nProcessing Training Data...")
for filename in os.listdir(train_dir):
    train_file_path = os.path.join(train_dir, filename)
    if os.path.isfile(train_file_path):
        train_text = load_text_from_file(train_file_path)
        if train_text:
            nlp(train_text)
            print(f"Processed training file: {filename}")

print("\nEvaluating on Test Data...")
all_extracted_keywords = [] # Store extracted keywords for all test files.
all_test_texts = [] # Store all test texts

all_precisions = []
all_recalls = []
all_f1s = []

for filename in os.listdir(test_dir):
    test_file_path = os.path.join(test_dir, filename)
    if os.path.isfile(test_file_path):
        test_text = load_text_from_file(test_file_path)
        if test_text:
            extracted_keywords_with_scores = extract_keywords_pytextrank(test_text)
            extracted_keywords = [kw[0] for kw in extracted_keywords_with_scores]

            all_extracted_keywords.extend(extracted_keywords) # Append to the global list
            all_test_texts.append(test_text) # Append to the global List

            precision, recall, f1 = evaluate_keywords(reference_keywords, extracted_keywords)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            print(f"File: {filename}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
            print(f"   Extracted: {extracted_keywords}")
            print(f"   Reference: {reference_keywords}")

if all_f1s:
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print(f"\nAverage Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1-score: {avg_f1}")
else:
    print("No test files processed.")

# Build and display the concept map after processing all test files
if all_extracted_keywords and all_test_texts:
    combined_text = " ".join(all_test_texts) # Combine all test texts
    build_concept_map(all_extracted_keywords, combined_text)