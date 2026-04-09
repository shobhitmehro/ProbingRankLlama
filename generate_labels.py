import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5Model
import torch
from collections import Counter
from scipy.spatial.distance import jaccard
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon


nltk.download('punkt')  # Tokenizer model
nltk.download('stopwords')  # Stopwords list
stop_words = set(stopwords.words('english'))




def tokenize_documents(documents):
    tokenized = []
    for document in documents:
        words = word_tokenize(document)
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        tokenized.append(filtered_words)
    return tokenized

def compute_metrics(query_set):
    results = {}
    
    for query, documents in query_set.items():

        documents_tokens = tokenize_documents(documents)
        query_tokens = tokenize_documents([query])[0]  # Tokenize query similarly

        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        
        # Compute TF statistics
        tf_docs = np.array(doc_term_matrix.toarray())
        tf_query = np.array(query_vector.toarray())
        
        # Document length (stream length)
        doc_lengths = tf_docs.sum(axis=1)
        
        # Covered query terms and ratios
        covered_query_terms = (tf_query > 0).sum()
        covered_query_term_ratio = covered_query_terms / len(vectorizer.get_feature_names_out())
        
        # Compute document IDF
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)
        tfidf_matrix = tfidf_vectorizer.fit_transform([query]+documents)
        doc_tfidf = tfidf_matrix[1:].toarray()
        
        # Stream length normalized TF
        norm_tf = tf_docs / doc_lengths[:, np.newaxis]

        # Initialize lists to store similarity scores
        tfidf_cosine_scores = []
        jaccard_scores = []
        bm25_scores = []
        embedding_cosine_scores = []
        euclidean_scores = []
        manhattan_scores = []
        kl_divergence_scores = []
        js_divergence_scores = []
        t5_cosine_scores = []

     
        # 1. TF-IDF Cosine Similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query] + documents)
        query_vector = tfidf_matrix[0:1]
        document_vectors = tfidf_matrix[1:]
        for doc_vector in document_vectors:
            cosine_sim_tfidf = cosine_similarity(query_vector, doc_vector)[0][0]
            tfidf_cosine_scores.append(cosine_sim_tfidf)


        # # 4. BERT Embedding Cosine Similarity, Euclidean Distance, Manhattan Distance
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # model = AutoModel.from_pretrained("bert-base-uncased")
        # def get_embeddings(text):
        #     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        #     outputs = model(**inputs)
        #     return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
        # query_embedding = get_embeddings(query)
        # document_embeddings = [get_embeddings(doc) for doc in documents]
        # for doc_embedding in document_embeddings:
        #     cosine_sim_embedding = cosine_similarity(query_embedding, doc_embedding)[0][0]
        #     embedding_cosine_scores.append(cosine_sim_embedding)            
        #     euclidean_dist = euclidean_distances(query_embedding, doc_embedding)[0][0]
        #     euclidean_scores.append(euclidean_dist)
        #     manhattan_dist = manhattan_distances(query_embedding, doc_embedding)[0][0]
        #     manhattan_scores.append(manhattan_dist)

        # 5. KL Divergence and Jensen-Shannon Divergence
        query_freq = Counter(query_tokens)
        all_words = list(set(query_tokens).union(*documents_tokens))
        for doc_tokens in documents_tokens:
            document_freq = Counter(doc_tokens)
            p = np.array([query_freq[word] / len(query_tokens) for word in all_words])
            q = np.array([document_freq[word] / len(doc_tokens) for word in all_words])            
            # KL Divergence
            kl_div = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            kl_divergence_scores.append(kl_div)
            # Jensen-Shannon Divergence
            js_div = jensenshannon(p, q) ** 2
            js_divergence_scores.append(js_div)

        # # 6. T5 Embedding Cosine Similarity
        # t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # t5_model = T5Model.from_pretrained("t5-base")
        # def get_t5_embeddings(text):
        #     inputs = t5_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        #     outputs = t5_model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        #     return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
        # query_t5_embedding = get_t5_embeddings(query)
        # document_t5_embeddings = [get_t5_embeddings(doc) for doc in documents]
        # for doc_t5_embedding in document_t5_embeddings:
        #     t5_cosine_sim = cosine_similarity(query_t5_embedding, doc_t5_embedding)[0][0]
        #     t5_cosine_scores.append(t5_cosine_sim)


        # Compute metrics for each document
        doc_metrics = []
        for i, doc in enumerate(documents):
            # a = covered_query_term_ratio
            # b = np.mean(norm_tf[i])
            # c = np.sum(tf_query > 0) #covered query term number
            # d= np.var(doc_tfidf[i])
            metrics = {
                # "covered query term number": np.sum(tf_query > 0),
                # "covered query term ratio": covered_query_term_ratio,
                # "stream length": doc_lengths[i],
                # "sum of term frequency": np.sum(tf_docs[i]),
                # "min of term frequency": np.min(tf_docs[i]),
                # "max of term frequency": np.max(tf_docs[i]),
                # "mean of term frequency": np.mean(tf_docs[i]),
                # "variance of term frequency": np.var(tf_docs[i]),
                # "sum of stream length normalized term frequency": np.sum(norm_tf[i]),
                # "min of stream length normalized term frequency": np.min(norm_tf[i]),
                # "max of stream length normalized term frequency": np.max(norm_tf[i]),
                # "mean of stream length normalized term frequency": np.mean(norm_tf[i]),
                # "variance of stream length normalized term frequency": np.var(norm_tf[i]),
                # "sum of tf*idf": np.sum(doc_tfidf[i]),
                # "min of tf*idf": np.min(doc_tfidf[i]),
                # "max of tf*idf": np.max(doc_tfidf[i]),
                # "mean of tf*idf": np.mean(doc_tfidf[i]),
                # "variance of tf*idf": np.var(doc_tfidf[i])
                # "combo1":a+b, 
                # "combo2":(a+b)*(a+b), 
                # "combo3":(a+b)*(a+b)*(a+b), 
                # "combo4":(d+a),
                # "combo5":(a+d)*(a+d),
                # "combo6":(a+d)*(a+d)*(a+d),
                # "combo7":a+b+c,
                # "combo8":b+c+d,
                # "combo9":a+c+d,
                # "combo10":(a+b+d),
                # "combo11":(a+b+d)*(a+b+d),
                # "combo12":(a+b+d)*(a+b+d)*(a+b+d)
                "tfidf_cosine_scores": np.array(tfidf_cosine_scores[i]),
                #"manhattan_scores": np.array(manhattan_scores[i]),
                "kl_divergence_scores": np.array(kl_divergence_scores[i]),
                "js_divergence_scores": np.array(js_divergence_scores[i]),
                # "BERT_cosine_scores": np.array(embedding_cosine_scores[i]),
                # "t5_cosine_scores": np.array(t5_cosine_scores[i])

            }
            #print(metrics)
            doc_metrics.append(metrics)
        
        # Compute BM25
  
       
        # Compute BM25
        bm25 = BM25Okapi(documents_tokens)
        bm25_scores = bm25.get_scores(query_tokens)
        
        
        # Add BM25 scores to metrics
        for i, score in enumerate(bm25_scores):
            doc_metrics[i]["BM25"] = score
        
        results[query] = doc_metrics

    return results

#Example Usage
# query_set = {
#     "What is AI?": ["Artificial Intelligence is the branch of engineering and science devoted to constructing machines that think.",
#                     "AI is the field of science which concerns itself with building hardware and software that replicates human functions such as learning and reasoning."],
#     "Explain machine learning": ["Machine learning is a type of artificial intelligence that enables self-learning from data and applies that learning without human intervention.",
#                                 "It is the scientific study of algorithms and statistical models that computer systems use to perform specific tasks."]
# }

# results = compute_metrics(query_set)
# print(results)
