from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime

from collections import Counter

DATABASE_URL = f'bolt://localhost:7687'
# USER = 'dani'
USER = 'neo4j'
PASSWORD = 'admin123'

driver = GraphDatabase.driver(DATABASE_URL, auth=(USER, PASSWORD))


# Import the projection of the graph with the important information for the task

query = """
MATCH (k:Keyword)<-[]-(p:Paper)-[]->(a:Author)-[]->(o:Organization)
RETURN  distinct *
"""

results = driver.session().run(query)

# Load the neo4j graph to a networkx graph
G = nx.MultiDiGraph()

nodes = list(results.graph()._nodes.values())
for node in nodes:
    G.add_node(node.element_id, labels=list(node._labels)[0], properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, type=rel.type,
               properties=rel._properties)

# Map the type with the generated nodes of the networkx graph

rev_graph_dict = {}
paper_graph_dict = {}
for node in G.nodes():
    if G.nodes[node]['labels'] == 'Author':
        rev_graph_dict[node] = G.nodes[node]['properties']['name']

    if G.nodes[node]['labels'] == 'Paper':
        paper_graph_dict[node] = G.nodes[node]['properties']['title']

# Generate node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, p=1, q=1)
model = node2vec.fit(window=10, min_count=1)

# Prepare the interactions
papers = []
reviewers = []
for paper in paper_graph_dict.keys():
    for reviewer in rev_graph_dict.keys():
        papers.append(paper)
        reviewers.append(reviewer)

# Prepare the data using the embeddings
X_reviewer = np.array([model.wv[node] for node in reviewers])
X_paper = np.array([model.wv[node] for node in papers])

# Do the predictions
loaded_model = load_model('model/reviewer_recommender.h5')
predictions = loaded_model.predict([X_reviewer, X_paper])
predictions = np.round(predictions).flatten()

df_predictions = pd.DataFrame({'Paper': papers,
                               'Reviewer': reviewers,
                               'Recommended': predictions})

df_predictions['Paper'] = df_predictions['Paper'].apply(lambda x: paper_graph_dict[x])
df_predictions['Reviewer'] = df_predictions['Reviewer'].apply(lambda x: rev_graph_dict[x])
df_predictions['Recommended'] = df_predictions['Recommended'].apply(lambda x: 'Yes' if x == 1.0 else 'No')

# Select only recommended reviewers
df_predictions = df_predictions[df_predictions['Recommended'] == 'Yes']

# Export DataFrame to CSV
current_time = datetime.now().strftime("%d%m%YT%H%M%S")
output_filename = f'recommendations/predictions_{current_time}.csv'
df_predictions.to_csv(output_filename, index=False)
