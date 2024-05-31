# The Sound of Links

## Abstract

Spotify’s recommender system is a key feature that enhances user experience by suggesting tracks to expand and improve new or existing playlists. This study explores the effectiveness of network analysis techniques for predicting links in Spotify’s playlist-track data. We first reduce the dataset by sub-sampling it and using a k-core subgraph method to manage computational complexity. We then compare the performance of traditional Machine Learning methods (KNN) applied to handmade network features with advanced Graph Neural Networks (GNNs) such as LightGCN, GAT, and SAGE. Additionally, we incorporate Spotify’s audio features as initial embeddings to improve the performance of our GNN models. Our findings reveal that while GNNs with audio features show improved predictions, the simpler KNN model using handmade network features surprisingly outperforms GNNs. This suggests that we don’t need complex models to get good link predictions, but can rely on the very powerful yet simple structural features of the network.

![alt text](https://github.com/BP4769/INA-Project-Spotify-Challange/blob/main/plots/Pipeline.png)

## Introduction

Spotify is a leading platform in the audio-streaming business, with over 600 million users. One of the features that attract many customers to Spotify is their recommender system. Although not all the details about their algorithms are known, Spotify is open about some of the methods they use (Collaborative and Content-Based filtering), and encourage the public to come up with solutions of their own. One such example was the RecSys Spotify 1 Million Playlist Challenge in 2018, where teams from around the world competed to create the best Playlist completion algorithm. Spotify provided the dataset of 1 million playlists created by users between 2010 and 2017, the metrics, and a challenge test dataset with a hidden ground truth. The competitors had to create 500 track predictions for each playlist in 10 different scenarios, with varying amounts of metadata and the number of tracks already included in the playlist.

After the official RecSys challenge was concluded, a continuation of the challenge was opened on AICrowd and is still running. The challenge is by itself an intriguing problem for any computer scientist, but our interests were spiked by the lack of Network Analysis based solutions in the top contenders of the official challenge. Since the music data falls so naturally into a graph shape and this is after all the Introduction to Network Analysis course, we set out to test how the network-based approaches would fare in the challenge.

## Methods

### Data

Our project uses the Spotify Million Playlist Dataset (MPD), which originally supported the RecSys Challenge 2018 and was re-released on AIcrowd in 2020, making it accessible for the development of music recommendation systems. The dataset includes one million real user-generated playlists created from 2010 to 2017. The collection consists of over two million unique tracks, which vary in terms of genre and other thematic groups. Each playlist includes the title, track list, and other meta information. Each track in a playlist consists of the unique track URI among other track information. Due to computational constraints, our analysis focuses on a subset of the MPD. We selected the first 50 JSON files from the dataset, representing 50,000 playlists, which is a manageable yet sufficiently diverse sample for our model training and testing in a standard Google Colab notebook with GPU RAM.

### Building a Graph

For the purpose of our network analysis, we imagined our data as a bipartite graph where one set of nodes represents playlists and the other set represents tracks. An edge between two nodes from different sets indicates the inclusion of the track in the playlist. Firstly, we defined classes for loading the data, which were Track, Playlist, and JSONFile. After loading the data, we built a bipartite graph, which contained 511,880 nodes and 3,300,244 edges. The data was too big to process in our pipeline, so we decided to make a reasonable subgraph of the data, which would reduce the set of nodes while maintaining a dense subgraph structure. We decided on making a k-core subgraph, where we chose k=30. The subgraph contained 34,753 nodes (22,527 playlists, 12,226 tracks) and 1,575,269 edges. The k-core reduction strategy effectively reduced the dataset’s size while still maintaining a large percentage of network edges. Since our analysis also focused on using Spotify audio features as initial embeddings, we added track features using Spotify API to each track. Each track has a numerical value between 0 and 1 (those that weren’t were normalized) for the next features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and time signature. We additionally applied standardization to the 12 track features with a mean of 0 and a standard deviation of 1. Afterwards, we created an expanded features matrix, which contained all pairwise interactions of features, including each feature with itself. We computed it through element-wise multiplication of two feature column vectors. We concatenated standardized track features with all pairwise interaction features to an expanded features matrix. We created a PyTorch Data object for our graph, which included the edge indices and the expanded features matrix for the node feature matrix.

### Train-val-test Split

We had to pick an appropriate graph split for transductive link prediction. We employed the PyTorch Geometric’s RandomLinkSplit function, designed to effectively perform splits that respect the graph’s topology without any data leakage. This split returns two types of edges: message passing edges and supervision edges. The message passing edges are used to propagate information across the network, while the supervision edges are the edges we aim to predict. The method randomizes the removal of edges to create training, validation, and test splits and ensures the following two conditions:

- The training set does not contain edges present in the validation and test sets.
- The validation set does not contain edges present in the test set.

### Models

For this project, we used four different models. For the baseline, we chose the KNN algorithm and ran it on the extracted network features which included: ID of the first node, ID of the second node, a train/test flag for splitting, the preferential attachment index, Jaccard index, Adamic-Adar index, and two community flags indicating if the two nodes were part of the same community according to the Louvain and Infomap algorithms.

Next, we employed three versions of the LightGCN algorithm with three different convolutional layers: LGN (default for LightGCN), GAT, and SAGE as seen in previous studies. All models were trained for 300 epochs with three convolutional layers, a learning rate of 0.01, and a weight decay of 1e-5. We used Bayesian Personalized Ranking (BPR) loss. All models were first trained with no initial embeddings with 78 dimensions as well as with the 78 Spotify Audio features that we created. To predict potential playlist-track pairs, we calculate the dot product between the two embeddings, which serves as a measure of similarity.

### Evaluation

We used AUC and accuracy of predictions to evaluate the models’ performance. We evaluated the GNN on training supervision edges, validation edges, and test edges, as described in the data split section.

## Results

Each model’s performance was assessed using ROC AUC and accuracy, with detailed visual representations of training and validation phases of our GNN models presented. Our results indicate that initializing embeddings with track features significantly improved the models’ performance metrics with LGC and GAT convolutional layers. However, the SAGE model exhibits signs of overfitting, especially when augmented with additional track features. Overfitting can be seen in the ROC AUC and accuracy on the train set reach nearly perfect scores, while the scores on the validation set drop significantly.

Evaluation scores of the test set are shown in Table 1 for all our models. Our baseline KNN algorithm which utilized handmade network features performed surprisingly well given its computational simplicity. It outperformed all our GNN models in terms of both ROC AUC and accuracy. The performance metrics of GNN models highlight the improved performance when the models incorporate track features.

| Model    | Spotify Features | ROC AUC | Accuracy |
|----------|-------------------|---------|----------|
| LGC      | No                | 0.8807  | 0.6455   |
| LGC      | Yes               | 0.8985  | 0.6842   |
| GAT      | No                | 0.7385  | 0.6603   |
| GAT      | Yes               | 0.8923  | 0.8178   |
| SAGE     | No                | 0.7070  | 0.6625   |
| SAGE     | Yes               | 0.6876  | 0.6408   |
| Baseline | No                | 0.9207  | 0.8402   |

## Discussion

We provide several insights into the use of network analysis for link prediction on Spotify playlist-track data. Firstly, when building a graph, the application of a k-core reduction strategy effectively reduced the graph size, while still maintaining a high percentage of network links. This approach was fundamental for our further analysis and management of the computational demands of GNNs.

Next, augmenting original data with Spotify audio features and using them as initial embeddings in GNNs significantly boosted their predictive performance. These results confirmed our hypothesis that domain-specific information can fundamentally improve recommendation systems’ performance. However, the incorporation led to overfitting in our GNN SAGE pipeline, which calls for caution in complex GNN architectures.

The robust performance of our baseline KNN model was a surprising key finding of our study, which undermines the common assumption that more complex models yield better results. Our results suggest that for certain types of networks and network-based predictions, simpler models not only provide lower computational demands but can also be preferable in terms
