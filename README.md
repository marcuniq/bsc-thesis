# Bachelor Thesis
## Heterogeneous Information Sources for Recommender Systems

### Abstract
The most popular algorithm for recommender systems utilizes the collaborative filtering technique which makes only use of the user-item rating matrix. This thesis introduces two approaches which employ extra data encoded as feature vectors. One of our proposed models, MPCFs-SI, is based on a nonlinear matrix factorization model for collaborative filtering (MPCFs) and utilizes the extra data to regularize the model. The second model called MFNN is an ensemble of a matrix factorization and a neural network and uses the extra data as an input to the neural network. Our results show that MPCFs-SI outperforms the baseline recommender MPCFs on a subset of both MovieLens 100k and MovieLens 1M datasets. MFNN is inferior to the MPCFs model on our MovieLens 100k subset, however, it is at a similar performance level as MPCFs-SI on the bigger MovieLens 1M subset.


My learnings:
* improved knowledge of technologies and concepts (Python, Machine Learning, Recommender Systems, Neural Networks)
* got to know new technologies and concepts (Collaborative Filtering, Matrix Factorization, Word2Vec/Doc2Vec)
* in practice, use SLIM[1] for very fast and good enough results, or use MPCFs[2] for better results

[1]: Ning, X. and Karypis, G. (2011). SLIM: Sparse Linear Methods for top-N recommender systems

[2]: Kabbur, S. (2015). Machine Learning Methods for Recommender Systems

The bachelor thesis can be downloaded here: [thesis.pdf](/thesis/thesis.pdf)

