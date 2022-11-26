import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import operator
import re

from networkx.algorithms.community import greedy_modularity_communities
import networkx.algorithms.community as nx_comm

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

####

st.title('Internet Submarine Cables Network')

st.markdown("## Introduction")

st.markdown("The idea of this project was born following what happened in Egypt, along the Suez Canal. The running aground of the ship Ever Given, which lasted about a week, has literally blocked the transit of one of the most important nodes of maritime trade.")
st.markdown("Industry and trade-related sectors suffered heavy losses, exacerbating an already fragile situation due to the effects of the pandemic. On the contrary, the services sector, where we can find more **intangible** services, may be less affected by events of this magnitude, but this does not mean that they are exempt from risks, aka **black swans**, unexpected events whose consequences may be decidedly significant.")
st.markdown("What would be a black swan for the services sector? In my opinion its 'fuel' is provided by continuous technological progress, by an increasingly *interconnected* and *real time* society: what would happen if this connection 'physically' were to break down?  \n As can be read [here](https://en.wikipedia.org/wiki/Submarine_communications_cable), some event of this type already took place in the past years, with (luckly) not so significative damages.")

st.markdown("The network that allows internet connection is physically provided by submarine cables that allow data transfer through fiber optics. These cables are born and connected on land, in precise locations, from which then, through the most standard connections, arrive in homes and offices as we are accustomed.")

st.markdown("Having said that, the question I asked myself is: \n - What are the most important arrival/departure points for these cables?")

st.markdown("## 1.0 Data Retrival")

st.markdown("In order to answer the previous question, I retrieved two datasets from the following [source](https://hub.arcgis.com/datasets/Story::telecomcables-2018?geometry=-174.023%2C-66.926%2C174.024%2C82.357&selectedAttribute=Shape__Length), where there are informations related to: \n - submarine cables \n - cities affected by each cables.")
st.markdown("In a hypothetical more in-depth job, this data retrieval step can be automated using a simple *scraper*. Let's have a look at the two datasets, before merging them into a final one.")

####

city = pd.read_csv('Submarine_Cables_and_Terminals__2018_city.csv')
cable = pd.read_csv('Submarine_Cables_and_Terminals__2018_cable.csv')
city = city.drop(columns='OBJECTID')
cable = cable.drop(columns=['OBJECTID','Shape__Length'])
city = city.rename(columns={'Y':'latitude','X':'longitude'})

st.markdown("Cities dataset:")
st.table(city.head(1))
st.markdown("Cables dataset:")
st.table(cable.head(1))

df = pd.merge(city,cable,on='cable_id',suffixes=['_city','_cable'],how='inner')
df = df.drop_duplicates()
df2 = pd.DataFrame(df.groupby('cable_id')['Name_city'].unique())

st.markdown("## 2.0 Data Preparation")

st.markdown("In order to find the correct relatioship between cities bounded by the same cables, i've created a **list of adjacencies** useful later for the creation of the network.")

df4 = []
for i in range(df2.shape[0]):
    coppie = df2.iloc[i].Name_city.tolist()
    df3 = pd.DataFrame([(p1, p2) for p1 in coppie for p2 in coppie if p1 != p2],columns=['source','target'])
    df3['cable_id'] = int(df2.iloc[i].name)
    df4.append(df3)
df4 = pd.concat(df4)
df5 = pd.merge(df4,df[['cable_id','Name_cable','length','ReadyForServiceDate','owners','Name_city']],left_on=['cable_id','source'],right_on=['cable_id','Name_city'],how='left')
df5 = df5.drop(columns=['Name_city'])

st.table(df3.head(3))

st.markdown("In order to retrive all the informations available from the original dataset i've decided to: \n - edit the 'length' field and transform it into a numerical --> this could be a weight for a weighted graph \n - retrive information about how many cities every cable 'touches': the more it touches, the more important is the cable --> this could be an attribute to the edges \n - retrive information about how many different cables touch every city: the more a city is reached by different cable the most important it should be --> this could be an attribute to the nodes")

df5['length'] = df5['length'].str.replace(r'\D','')
df5['length'] = pd.to_numeric(df5['length'])
df5['freq_cable'] = df5.cable_id.map(df5.groupby('cable_id')['source'].nunique())
df5['freq_city'] = df5.source.map(df5.groupby('source')['target'].nunique())
df6 = pd.merge(df5,city,left_on=['source','cable_id'],right_on=['Name','cable_id'],how='left')

st.markdown("## 3.0 Modelling")

st.markdown("Let's start by building the simple network structures. I've take the following decison in order to answer my original question: which cities are more important for the internet connection? \n - graph type: undirected \n - nodes: cities as start and endpoint of the cables \n - edges: cables between cities")
st.code("g = nx.from_pandas_edgelist(df6,source='source',target='target',edge_attr=True, create_using=nx.Graph())")
g = nx.from_pandas_edgelist(df6,source='source',target='target',edge_attr=True, create_using=nx.Graph())

st.markdown("### 3.1 Network Evaluation")

st.markdown("Having given this structure what emerges from an initial analysis?")
st.text(nx.info(g))
st.markdown("From a first look emerges that the average degree is segnaling us that every node as in mean around 12 link: every city in the dataset has around 12 cables connections.  \n However the mean could not be the right metric to use: let's have a look at the degree distribution.")

degree = [g.degree(n) for n in g.nodes()]
st.image("bar_chart.png")

st.markdown("The degreee distribution reflects one of the typical element found in most of other **real** networks: asymmetric and fat-tailed distribution, leading to **hubs**, few and important nodes.")
st.markdown("What about **density**?")
st.text(nx.density(g))
st.markdown("This value said that the proportion between nodes and link is highly skewed: the density is low.")
st.markdown("What about che connectivity between the nodes? Let's see the **average clustering coefficient**.")
st.text(nx.average_clustering(g))
st.markdown("This value tells us that the nodes in the network are quite connected to each other.")

st.markdown("Having concluded this phase of macro analysis of the network, we can draw some considerations: \n - we are faced with a network in which the nodes are well connected to each other \n - we are faced with a network whose links are concentrated in some subset of nodes, given the low density value and the presence of significant hubs.  \n At this point we are going to analyze the **centrality** of the nodes to identify those most important for the entire network, those on which it is good to pay attention to maintain a positive health of the Internet network.")

st.markdown("### 3.2 Node Evaluation")

st.markdown("#### 3.2.1 Degree Centrality")
degc = nx.degree_centrality(g)
st.dataframe(sorted(degc.items(), key=operator.itemgetter(1), reverse=True)[0:5])
st.markdown("From the degree centrality results is visible how the top 5 nodes are all located in the area around the Indian Ocean, from Saudi Arabia to India. The nodes in this area presents the highest number of link connected.")

st.markdown("#### 3.2.2 Betweenness Centrality")
degc = nx.betweenness_centrality(g)
st.dataframe(sorted(degc.items(), key=operator.itemgetter(1), reverse=True)[0:5])
st.markdown("From the betweenness centrality results is visible instead which are the most important nodes in terms of network connection, i.e. the nodes that act as connectors for other nodes, lying on a large number of paths (cables). It's interesting to note that the most important node in this section lies in Italy and it can be seen as a connector between the Mediterranean Sea cables.")

st.markdown("#### 3.2.3 Closeness Centrality")
degc = nx.closeness_centrality(g)
st.dataframe(sorted(degc.items(), key=operator.itemgetter(1), reverse=True)[0:5])
st.markdown("Also from the closeness centrality results is visibile that the *Mazara del Vallo* node is important in terms of proximity to other nodes. After this italian node is visibile (again) the nodes of the Indian Ocean area, as seen previously in the degree centrality.")

st.markdown("#### 3.2.4 Eigenvector Centrality")
degc = nx.eigenvector_centrality(g)
st.dataframe(sorted(degc.items(), key=operator.itemgetter(1), reverse=True)[0:5])

st.markdown("#### 3.2.5 PageRank")
degc = nx.pagerank(g)
st.dataframe(sorted(degc.items(), key=operator.itemgetter(1), reverse=True)[0:5])
st.markdown("As for the spectral centrality metrics, thus based on the eigenvalues of the matrices, I used the classic eigenvector centrality and its normalized version (PageRank) to determine the influence of each node (city) in the network. Both results confirm that the most important nodes are found in the area of the Indian Ocean, between Saudi Arabia and India. ")

st.markdown("#### 3.2.6 Bivariate Analysis")
connectivity = list(g.degree())
connectivity_values = [n[1] for n in connectivity]
centrality = list(nx.eigenvector_centrality(g).values())
st.image("bivariate.png")
st.markdown("Let's see who are the nodes on the right of the graph: these seems an important cluster by looking at their degree and centrality values. They can be renamed as the **influencer** nodes.")
deg = pd.DataFrame(list(g.degree()))
deg = deg.sort_values(by=1,ascending=False)
degc = nx.eigenvector_centrality(g)
degc = pd.DataFrame(degc.items())
val = pd.merge(deg,degc,on=0)
val.columns = ['city','degree','centrality']
influencer = val[(val['degree']>50) & (val['centrality']>0.01)]
st.table(influencer)
st.markdown("What emerged from comparing the centrality and degree of each node?   \n From this bivariate comparison, a clear division emerges between the most central and connected nodes and the least. From the influencer table above emerges also a peculiarity in my opinion: among the most influential nodes we do not find any point belonging to the American continent. From this analysis to dominate the network of the Internet are nodes belonging to the Mediterranean Sea basin and Southeast Asia.")

st.markdown("### 3.3 Communities Detection")
st.markdown("At this point it was deemed interesting to perform a community detection exercise, in order to identify that set of similar nodes (cities). In this case it could be mainly a matter of identifying **explicit communities**, dictated for example by being on the route of a single cable.")
c = list(greedy_modularity_communities(g))
st.markdown("How many communities has been found?")
st.text(len(c))
st.markdown("What is the value of **modularity** of the communities?")
st.text(nx_comm.modularity(g,c))
st.markdown("The optimization algorithm satisfactorily manages to partition the network with a decidedly positive modularity value. Now let's try to analyze how this communities are composed.  \n Indexes in the below dataframe represents the *communities_id* identified by the optimizator.")

comm = pd.DataFrame(c)
comm['community_member'] = 0

for i in range(comm.shape[0]):
    comm['community_member'].iloc[i] = i

comm_2 = comm.T.unstack().to_frame().sort_index(level=1)
comm_3 = comm_2.reset_index()
comm_3 = comm_3.drop(columns='level_1')
comm_3.columns = ['community_member','Name']

df7 = pd.merge(df6,comm_3,on='Name',how='left')
st.dataframe(df7.groupby('community_member')['length','freq_cable','freq_city'].median())

st.markdown("With this simple group-by with respect to the community_id identified by the algorithm, the operation of the optimizer is highlighted. This, in fact, groups the most important nodes, as can be seen from the median values of the fields 'freq_cable' and 'freq_city', significantly higher in the first communities than in the last ones.  \n Finally, it has been decided to enrich the dataset with the information derived from the biavariate analysis: the cluster of nodes with a high value of degree and centrality has been labeled as influencer. This informtion could be useful in the next graphical phase.")

influencer['influ'] = 'yes'
df8 = pd.merge(df7,influencer,left_on='Name',right_on='city',how='left')

st.markdown("## 4.0 Network Plot")
st.markdown("Worldwide Submarine cables and cities Network")
st.image("map_1.png",width=900)
st.markdown("Below the same plot with another background for better identifing some cliques.")
st.image("map_2.png",width=900)
st.markdown("In the above chart, I have highlighted the following information: \n * Nodes: \n    - the size is determined by the number of cables passing through that node \n    - the color is yellow for the nodes that the community detection has highlighted as belonging to the most numerous and connected cluster, in blue the nodes labeled as *influencer* from the bivariate analysis; red for the others. \n * Links: \n    - the thickness of the link is determined by the number of cities through which that link passes: the *thicker* the line, the more the link encounters different cities.")
st.markdown("From this first glance, what was previously reported by observing the various measures of centrality emerges: the most important nodes of this network all reside in Southeast Asia and Europe.  \n Now let's take a deeper look inside Europe.")
st.image("map_3.png")
st.markdown("In the map above is interesting to highlight the numerosity of yellow and blue nodes in the area of the Mediterranea Sea. Moreover, in the north of England is highlithed a particular cluster, composed by a lot of not important nodes.")

st.markdown("## 5.0 Conclusion & further analysis")
st.markdown("The aim of this work was to highlight which nodes are the most important for the network of undersea cables that allows the connection to the Internet on a global scale.  \n From this first analysis it has emerged that the main nodes are located between Europe and Southeast Asia, with cables that run along the seabed of the Indian Ocean and arrive in the Mediterranean Sea. Consequently, creating a system to monitor these nodes could be useful to efficiently oversee the global Internet network.")

st.markdown("The weaknesses of this project leave the door open for further and more in-depth analysis: \n - enrich the dataset with more relevant information (data throughput, speed, adverse events) \n - build predictive models able to classify the nodes most at risk, and so on. \n - build scenario analysis, in order to be ready to act if something occur in the future (**black swan**)")

st.markdown("**Note**: the graphical visualization offered is not faithful to the actual path of the nodes, which can be consulted at the following [link](https://www.submarinecablemap.com/) in an updated version: the dataset that it has been used was updated lastly only in 2018.")

st.markdown("Thanks for your attention  \n Daniele D'Avino")
