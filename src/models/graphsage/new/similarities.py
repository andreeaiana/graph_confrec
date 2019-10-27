# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

# DISCLAIMER:
# This code file is derived from
# https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec
# and
# https://tedboy.github.io/nlps/_modules/gensim/matutils.html


class Similarities():

    def __init__(self, embeddings, id_map):
        self.embeddings = embeddings
        self.id_map = id_map
        self._l2_normalize()

    # Function code derived from
    # https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.similar_by_vector
    def similar_by_vector(self, vector, topn=10):
        """
        Find the top-N most similar nodes by vector.

        If topn is False, similar_by_vector returns the vector of similarity
        scores.
        """
        return self.most_similar(positive=[vector], topn=topn)

    # Function code derived from
    # https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.most_similar
    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively
        towards the similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the
        projection weight vectors of the given nodes and the vectors for each
        node in the model.

        If topn is False, most_similar returns the vector of similarity scores.
        """
        # Allow calls like most_similar(vector), as a shorthand for
        # most_similar([vector])
        if isinstance(positive, np.ndarray):
            positive = [positive]
        if isinstance(negative, np.ndarray):
            negative = [negative]

        # Add weights for each word, if not already present;
        # default to 1.0 for positive and -1.0 for negative words
        positive = [(vector, 1.0) if isinstance(vector, np.ndarray) else
                    vector for vector in positive]
        negative = [(vector, -1.0) if isinstance(vector, np.ndarray) else
                    vector for vector in negative]

        # Compute the weighted average of all words
        all_vectors = set()
        mean = []
        for vector, weight in positive + negative:
            if vector in self.embeddings:
                idx = np.where(self.embeddings == vector)[0][0]
                mean.append(weight*self.normalized_embeddings[idx])
                all_vectors.add(idx)
            elif isinstance(vector, np.ndarray):
                mean.append(weight*vector)
            else:
                raise ValueError("Problem with vector %s" % vector)
        if not mean:
            raise ValueError("Cannot compute similarity with no input.")

        mean = self._unitvec(np.array(mean).mean(axis=0))
        dists = np.dot(self.normalized_embeddings, mean)

        if not topn:
            return dists

        best = self._argsort(dists, topn=topn+len(all_vectors), reverse=True)

        # Ignore (don't return) nodes from the input
        result = [(self.id_map[sim], float(dists[sim])) for sim in best if sim
                  not in all_vectors]
        return result[:topn]

    # Function code derived from
    # from https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.init_sims
    def _l2_normalize(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep
        the normalized ones (saves memory)
        """
        print("Precomputing L2-norms of node weight vectors.")
        if replace:
            for i in range(self.embeddings.shape[0]):
                self.embeddings[i, :] /= np.linalg.norm(
                        self.embeddings[i, :])
            self.normalized_embeddings = self.embeddings
        else:
            self.normalized_embeddings = self.embeddings / np.linalg.norm(
                    self.embeddings, 2, axis=1)[..., np.newaxis]

    # Function code derived from
    # https://tedboy.github.io/nlps/_modules/gensim/matutils.html#unitvec
    def _unitvec(self, vector):
        """
        Scale a vector to unit length. The only exception is the zero vector,
        which is returned back unchanged.
        """
        blas = lambda name, ndarray: sp.linalg.get_blas_funcs(
                (name,), (ndarray,))[0]
        blas_nrm2 = blas('nrm2', np.array([], dtype=float))
        blas_scal = blas('scal', np.array([], dtype=float))
        vector = np.asarray(vector, dtype=float)
        veclen = blas_nrm2(vector)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vector)
        else:
            return vector

    # Code function derived from
    # https://tedboy.github.io/nlps/_modules/gensim/matutils.html#argsort
    def _argsort(self, dists, topn=None, reverse=False):
        """
        Returns the indices of the `topn` smallest elements in array `dists`,
        in ascending order.
        If reverse is True, return the greatest elements instead,
        in descending order.
        """
        dists = np.asarray(dists)
        if topn is None:
            topn = dists.size
        if topn < 0:
            return []
        if reverse:
            dists = -dists
        if topn >= dists.size or not hasattr(np, "argpartition"):
            return np.argsort(dists)[:topn]
        most_extreme = np.argpartition(dists, topn)[:topn]
        return most_extreme.take(np.argsort(dists.take(most_extreme)))
