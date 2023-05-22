from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

DATABASE_URL = f'bolt://localhost:7687'
# USER = 'dani'
USER = 'neo4j'
PASSWORD = 'admin123'

driver = GraphDatabase.driver(DATABASE_URL, auth=(USER, PASSWORD))

# Get the information to label the data

query_reviewers_1 = """
MATCH (k:Keyword)<-[:RELATED_TO]-(p:Paper)
MATCH (p)-[:REVIEWED_BY]->(a:Author)
RETURN  distinct a.id as id, k.keyword as keyword
"""

query_reviewers_2 = """
MATCH (k:Keyword)<-[:RELATED_TO]-(p:Paper)
MATCH (p)-[:WRITTEN_BY]->(a:Author)
RETURN  distinct a.id as id, k.keyword as keyword
"""

query_papers = """
MATCH (k:Keyword)<-[:RELATED_TO]-(p:Paper)
RETURN  distinct p.id as id, k.keyword as keyword
"""


def resultToList(results):
    r_list = []
    for r in results:
        r_list.append(dict(r))
    return r_list


def foldResultToDict(r_list):
    r_dict = {}
    for item in r_list:
        key = item['id']
        keyword = item['keyword']
        if key in r_dict:
            r_dict[key].append(keyword)
        else:
            r_dict[key] = [keyword]
    return r_dict


results1 = resultToList(driver.session().run(query_reviewers_1))
results2 = resultToList(driver.session().run(query_reviewers_2))
reviewers_dict = foldResultToDict(results1 + results2)

results = resultToList(driver.session().run(query_papers))
papers_dict = foldResultToDict(results)

# Import the full graph

query = """
MATCH (n)-[r]->(c) RETURN distinct *
"""

# query = """
# MATCH (p:Paper)-[r:REVIEWED_BY]->(a:Author) RETURN  distinct *
# """

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

# Map the labels with the generated nodes of the networkx graph

rev_graph_dict = {}
paper_graph_dict = {}
for node in G.nodes():
    if G.nodes[node]['labels'] == 'Author':
        id = G.nodes[node]['properties']['id']
        if id in reviewers_dict:
            rev_graph_dict[node] = reviewers_dict[id]

    if G.nodes[node]['labels'] == 'Paper':
        id = G.nodes[node]['properties']['id']
        if id in papers_dict:
            paper_graph_dict[node] = papers_dict[id]

reviewers = []
papers = []
label = []
for kr, rev in rev_graph_dict.items():
    for kp, paper in paper_graph_dict.items():
        reviewers.append(kr)
        papers.append(kp)
        common_keyword = set(rev) & set(paper)
        label_val = 1 if common_keyword else 0
        label.append(label_val)



# Generate node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, p=1, q=1)
model = node2vec.fit(window=10, min_count=1)


# Prepare training data using the embeddings
X_reviewer = np.array([model.wv[node] for node in reviewers])
X_paper = np.array([model.wv[node] for node in papers])
y = np.array([label for label in label])

# Split the data into train and test sets
X_reviewer_train, X_reviewer_test, X_paper_train, X_paper_test, y_train, y_test = train_test_split(
    X_reviewer, X_paper, y, test_size=0.2, stratify=y, random_state=42
)

# Define the multi-input model architecture
reviewer_input = Input(shape=(128,))
paper_input = Input(shape=(128,))
concatenated = Concatenate()([reviewer_input, paper_input])
dense1 = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='sigmoid')(dense1)

# Create the model
model = Model(inputs=[reviewer_input, paper_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_reviewer_train, X_paper_train], y_train, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict([X_reviewer_test, X_paper_test])
y_pred = np.round(y_pred).flatten()
print(classification_report(y_test, y_pred, zero_division=1))
