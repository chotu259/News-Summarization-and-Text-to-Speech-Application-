from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from keybert import KeyBERT
kw_model= KeyBERT()
import numpy
nlp = spacy.load("en_core_web_sm")
def custom_serializer(obj):
    """Handles set and numpy.float64 types for JSON serialization."""
    if isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    if isinstance(obj, numpy.float64):
        return float(obj)  # Convert np.float64 to regular float
    raise TypeError(f"Type {type(obj)} not serializable")
def compare_all_articles(articles, sentiment_articles, summaries, relevant_topics, comparison_pipeline):
    coverage = {}
    for i, (article1, summary1, sentiment1, relevant1) in enumerate(zip(articles, summaries, sentiment_articles, relevant_topics)):
        for j in range(i + 1, len(summaries)):
            article2, summary2, sentiment2, relevant2 = articles[j], summaries[j], sentiment_articles[j], relevant_topics[j]
            common_relevant = relevant1 & relevant2

            # Generate comparison
            combined_text = f"Article {i}: {summary1}\nArticle {j}: {summary2}\nCompare the key points."
            comparison_text = comparison_pipeline(combined_text, max_length=100, min_length=20, do_sample=False)[0]['generated_text']

            # Impact Analysis
            if sentiment1 == "Positive" and sentiment2 == "Negative":
                impact_text = f"Article {i} boosts confidence, while Article {j} raises concerns."
            elif sentiment1 == "Negative" and sentiment2 == "Positive":
                impact_text = f"Article {i} raises concerns, but Article {j} reassures readers."
            elif sentiment1 == sentiment2 == "Positive":
                impact_text = f"Both articles create an optimistic outlook."
            else:
                impact_text = "Both highlight challenges, increasing uncertainty."

            coverage[f"comparison {i}"] = {"comparison": comparison_text, "impact": impact_text}
    return coverage

def extract_topics(text):
    topics = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), top_n=3)
    return [topic[0] for topic in topics]

def extract_named_entities(text):
    doc = nlp(text)
    entities = {ent.text for ent in doc.ents}
    return list(entities)

def extract_relevant_terms(text,model):
    words = set(extract_topics(text))
    entities = set(extract_named_entities(text))
    words.update(entities)

    # Encode the Article
    article_embedding = model.encode(text, convert_to_tensor=True)
    word_embeddings = model.encode(list(words), convert_to_tensor=True)
    similarities = cosine_similarity(article_embedding.reshape(1, -1), word_embeddings)[0]
    word_similarity_scores = dict(zip(list(words), similarities))
    top_5_words = sorted(word_similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return  {word for word, _ in top_5_words}
# this is set.


