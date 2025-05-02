import os
from tkinter import Tk
import pytextrank
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
import re
import networkx as nx
import matplotlib.pyplot as plt
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

NUM_KEYWORDS = 200
SPACY_MODEL = "en_core_web_md"

# Define data directories and files more explicitly
TRAIN_DATA_DIR = "./Train_data"
TEST_DATA_DIR =  "./Test_Data"
KEYWORDS_FILE =  "./Keywords.txt"
INDEX_BY_CHAPTER = "./index_by_chapter.txt"


def load_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_reference_keywords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return {lemmatizer.lemmatize(re.sub(r',\s*\d+$', '', line.strip().lower())) for line in f}
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return None

def extract_keywords_pytextrank(text, nlp, num_keywords):
    doc = nlp(text)
    textrank_keywords = {}
    stopwords = nlp.Defaults.stop_words.union({"introduction", "conclusion", "however", "therefore", "in addition", "also", "furthermore"})

    # Extract keywords from TextRank and store with their rank
    for phrase in doc._.phrases:
        cleaned = phrase.text.strip().lower()
        if not any(char in string.punctuation for char in cleaned):
            tokenized = nlp(cleaned)
            valid_tokens = [token.text for token in tokenized if not token.is_stop and not token.is_punct and token.text not in stopwords]
            if valid_tokens:
                lemmatized = lemmatizer.lemmatize(" ".join(valid_tokens))
                textrank_keywords[lemmatized] = phrase.rank

    noun_chunks = {}
    # Extract and filter noun chunks
    for chunk in doc.noun_chunks:
        cleaned_chunk = chunk.text.strip().lower()
        tokenized_chunk = nlp(cleaned_chunk)
        valid_chunk_tokens = [token.text for token in tokenized_chunk if not token.is_stop and not token.is_punct and token.text not in stopwords]
        if valid_chunk_tokens and 1 < len(valid_chunk_tokens) <= 4: # Consider 2-4 word noun chunks
            lemmatized_chunk = lemmatizer.lemmatize(" ".join(valid_chunk_tokens))
            noun_chunks[lemmatized_chunk] = noun_chunks.get(lemmatized_chunk, 0) + 1 # Simple frequency count

    # Prioritize and Weight
    final_keywords = {}
    for kw, rank in textrank_keywords.items():
        final_keywords[kw] = rank * 1.2 # Slightly lower TextRank base weight

    for kw, freq in noun_chunks.items():
        if kw in final_keywords:
            final_keywords[kw] += 1.0 * freq # Increase weight for overlapping
        else:
            final_keywords[kw] = 0.4 * freq # Lower weight if only a noun chunk

    # Sort by weight and take top N
    sorted_keywords = sorted(final_keywords.items(), key=lambda item: item[1], reverse=True)
    return [kw for kw in sorted_keywords[:num_keywords]]

def get_vector(nlp, text):
    doc = nlp(text)
    vectors = [token.vector for token in doc]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, nlp.vocab.vectors_length))


def evaluate_with_embeddings(nlp, reference, predicted, similarity_threshold=0.6):
    tp = 0
    reference_matched = [False] * len(reference)
    predicted_matched = [False] * len(predicted)

    for i, pred_kw in enumerate(predicted):
        pred_vec = get_vector(nlp, pred_kw)
        for j, ref_kw in enumerate(reference):
            ref_vec = get_vector(nlp, ref_kw)
            if np.any(pred_vec) and np.any(ref_vec):
                # Ensure both vectors are 2D
                if ref_vec.ndim == 1:
                    ref_vec = ref_vec.reshape(1, -1)
                similarity = cosine_similarity(pred_vec, ref_vec)[0][0]
                if similarity >= similarity_threshold and not reference_matched[j]:
                    tp += 1
                    reference_matched[j] = True
                    predicted_matched[i] = True
                    break

    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(reference) if reference else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def evaluate_model(test_results, reference_keywords_by_chapter, nlp, similarity_threshold=0.6):
    if not reference_keywords_by_chapter:
        print("Reference keywords by chapter not loaded. Skipping chapter-based evaluation.")
        return

    chapter_precisions = {}
    chapter_recalls = {}
    chapter_f1s = {}
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for filename, result in test_results.items():
        predicted_keywords_with_scores = result["extracted_keywords"] # This seems to be a list of tuples
        predicted_keywords = [kw for kw, score in predicted_keywords_with_scores] # Extract just the keywords
        chapter = result["chapter"]
        if chapter and chapter in reference_keywords_by_chapter:
            reference_keywords = reference_keywords_by_chapter[chapter]
            precision, recall, f1 = evaluate_with_embeddings(nlp, list(reference_keywords), predicted_keywords, similarity_threshold=similarity_threshold)
            chapter_precisions.setdefault(chapter, []).append(precision)
            chapter_recalls.setdefault(chapter, []).append(recall)
            chapter_f1s.setdefault(chapter, []).append(f1)
            print(f"\nEvaluation for {filename} (Chapter {chapter}):")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    print("\n--- Chapter-wise Evaluation Summary (with Embeddings, Threshold={:.1f}) ---".format(similarity_threshold))
    for chapter in chapter_precisions:
        all_precisions.extend(chapter_precisions[chapter])
        all_recalls.extend(chapter_recalls[chapter])
        all_f1s.extend(chapter_f1s[chapter])

    overall_avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    overall_avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    overall_avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0

    print("Overall Average:")
    print(f"  Average Precision: {overall_avg_precision:.4f}")
    print(f"  Average Recall: {overall_avg_recall:.4f}")
    print(f"  Average F1-score: {overall_avg_f1:.4f}")


def process_data(directory_path, nlp):
    results = {}
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            text = load_text(filepath)
            if text:
                all_extracted_keywords = extract_keywords_pytextrank(text, nlp, num_keywords=NUM_KEYWORDS)
                # Assuming chapter can be inferred from the filename (e.g., ch1.txt)
                chapter = filename.replace("ch", "").replace(".txt", "")
                results[filename] = {
                    "extracted_keywords": all_extracted_keywords,
                    "all_extracted_keywords": all_extracted_keywords,
                    "text": text,
                    "chapter": chapter
                }
                print(f"Processed: {filename}, Extracted Keywords: {len(all_extracted_keywords)}")
    return results


def build_concept_map(keywords, nlp, similarity_threshold=0.6):
    G = nx.Graph()
    unique_keywords = list(set(keywords))
    keyword_vectors = {}
    valid_keywords = []

    print(f"Number of unique keywords: {len(unique_keywords)}")

    # Get spaCy vectors for each unique keyword, only if in vocabulary and has a vector
    for keyword in unique_keywords:
        if keyword in nlp.vocab and nlp.vocab[keyword].has_vector:
            vector = nlp.vocab[keyword].vector
            if not np.all(vector == 0) and not np.any(np.isnan(vector)):
                keyword_vectors[keyword] = vector.reshape(1, -1)
                G.add_node(keyword)
                valid_keywords.append(keyword)
                print(f"Added node: {keyword}")

    print(f"Number of nodes in the graph: {G.number_of_nodes()}")

    num_edges = 0
    edges = []
    edge_weights = []
    for i, keyword1 in enumerate(valid_keywords):
        for j, keyword2 in enumerate(valid_keywords[i+1:]):
            if keyword1 in keyword_vectors and keyword2 in keyword_vectors:
                similarity = cosine_similarity(keyword_vectors[keyword1], keyword_vectors[keyword2])[0][0]
                if similarity >= similarity_threshold:
                    G.add_edge(keyword1, keyword2, weight=similarity)
                    edges.append((keyword1, keyword2))
                    edge_weights.append(similarity)
                    num_edges += 1
                    print(f"Added edge between '{keyword1}' and '{keyword2}' with similarity: {similarity:.2f}")

    print(f"Number of edges in the graph: {G.number_of_edges()}")

    # --- Visualization ---
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        pos = nx.spring_layout(G, k=0.3, iterations=50)

        node_sizes = [2000 for _ in G.nodes()]
        node_colors = 'lightgreen'
        cmap = plt.cm.viridis

        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize edge weights to the range [0, 1]
        norm = plt.Normalize(vmin=min(edge_weights) if edge_weights else 0, vmax=max(edge_weights) if edge_weights else 1)
        mapped_edge_colors = cmap(norm(edge_weights))

        edge_collection = nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=mapped_edge_colors,
                                               width=5, alpha=0.7, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # For older versions of matplotlib
        cbar = fig.colorbar(sm, ax=ax, label='Semantic Similarity')

        ax.set_title("Concept Map", fontsize=16)
        ax.axis('off')
        fig.tight_layout()
        plt.show()
    else:
        print("No valid nodes or edges to draw the concept map.")

def build_concept_map_semantic(keywords, nlp, similarity_threshold=0.6):
    G = nx.Graph()
    unique_keywords = list(set(keywords))
    keyword_vectors = {}
    valid_keywords = []

    print(f"Number of unique keywords: {len(unique_keywords)}")

    # Get spaCy vectors for each unique keyword, only if in vocabulary and has a vector
    for keyword in unique_keywords:
        if keyword in nlp.vocab and nlp.vocab[keyword].has_vector:
            vector = nlp.vocab[keyword].vector
            if not np.all(vector == 0) and not np.any(np.isnan(vector)):
                keyword_vectors[keyword] = vector.reshape(1, -1)
                G.add_node(keyword)
                valid_keywords.append(keyword)
                print(f"Added node: {keyword}")

    print(f"Number of nodes in the graph: {G.number_of_nodes()}")


    num_edges = 0
    edges = []
    edge_weights = []
    for i, keyword1 in enumerate(valid_keywords):
        for j, keyword2 in enumerate(valid_keywords[i+1:]):
            if keyword1 in keyword_vectors and keyword2 in keyword_vectors:
                similarity = cosine_similarity(keyword_vectors[keyword1], keyword_vectors[keyword2])[0][0]
                if similarity >= similarity_threshold:
                    G.add_edge(keyword1, keyword2, weight=similarity)
                    edges.append((keyword1, keyword2))
                    edge_weights.append(similarity)
                    num_edges += 1
                    print(f"Added edge between '{keyword1}' and '{keyword2}' with similarity: {similarity:.2f}")

    print(f"Number of edges in the graph: {G.number_of_edges()}")

    # --- Visualization ---
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        pos = nx.spring_layout(G, k=0.3, iterations=50)

        node_sizes = [2000 for _ in G.nodes()]
        node_colors = 'lightgreen'
        cmap = plt.cm.viridis

        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize edge weights to the range [0, 1]
        norm = plt.Normalize(vmin=min(edge_weights) if edge_weights else 0, vmax=max(edge_weights) if edge_weights else 1)
        mapped_edge_colors = cmap(norm(edge_weights))

        # Draw the edges with the mapped colors
        edge_collection = nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=mapped_edge_colors,
                                               width=5, alpha=0.7, ax=ax)

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)

        # Draw the labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # For older versions of matplotlib
        cbar = fig.colorbar(sm, ax=ax, label='Semantic Similarity')

        ax.set_title("Concept Map", fontsize=16)
        ax.axis('off')
        fig.tight_layout()
        plt.show()
    else:
        print("No valid nodes or edges to draw the concept map.")

def load_index_by_chapter(filepath):
    chapter_keywords = {}
    current_chapter = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Chapter"):
                    current_chapter = line.split()[1]
                    chapter_keywords[current_chapter] = set()
                elif current_chapter and line:
                    keyword = lemmatizer.lemmatize(line.lower())
                    chapter_keywords[current_chapter].add(keyword)
    except Exception as e:
        print(f"Error loading index by chapter from {filepath}: {e}")
        return None
    return chapter_keywords


if __name__ == "__main__":
    nlp = spacy.load(SPACY_MODEL)
    nlp.add_pipe("textrank")

    index_by_chapter = load_index_by_chapter(INDEX_BY_CHAPTER)
    if index_by_chapter:
        print("\nLoaded Index by Chapter:")
        for chapter, keywords in index_by_chapter.items():
            print(f"  Chapter {chapter}: {len(keywords)} keywords")
    else:
        print("\nCould not load index by chapter.")

    # print("\nProcessing Training Data...")
    # train_results = process_data(TRAIN_DATA_DIR, nlp)

    print("\nProcessing Test Data...")
    test_results = process_data(TEST_DATA_DIR, nlp)

    print("\nEvaluating Model on Test Data (against reference keywords by chapter)...")
    evaluate_model(test_results, index_by_chapter, nlp, similarity_threshold=0.6)

    all_extracted_test_keywords = [res["extracted_keywords"] for res in test_results.values()]
    combined_keywords = [kw for sublist in all_extracted_test_keywords for kw, score in sublist] 
    if combined_keywords and nlp:
        build_concept_map(combined_keywords, nlp)
   