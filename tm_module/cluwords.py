from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Normalizer
import numpy as np
from tm_module.post_processing import save_topics, dominant_topics, clean_topics
from sklearn.decomposition import NMF as NMF_sklearn
import warnings
import time
from tm_module.topic_modeling import TopicModeling
from tm_module.utils.reader import Reader
from tm_module.utils.logger import Logger


class CluWords(TopicModeling):
    """CluWords class for generating topic representations using CluWords algorithm.

    Args:
        reader (Reader): The reader object for reading the input data.

    Attributes:
        logger (Logger): The logger object for logging messages.

    """

    def __init__(self, reader: Reader):
        super().__init__(reader)
        self.logger = Logger('CluWords').get_logger()

    def _build_embedding(self, embedding_file: str, embedding_bin: bool, data: list):
        """Build the word embedding model.

        Args:
            embedding_file (str): The path to the word embedding file.
            embedding_bin (bool): Whether the embedding file is in binary format.
            data (list): The input data for building the embedding.

        Returns:
            tuple: A tuple containing the words vector, the number of cluwords, and the percentage of vocabulary in the embedding.

        """
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=embedding_bin)
        vec = CountVectorizer()
        vec.fit(data)
        dataset_words = vec.get_feature_names_out()
        model_words = model.index_to_key
        intersection_vocab = set(dataset_words).intersection(model_words)
        words_vector = {word: model[word] for word in intersection_vocab}
        del model
        n_words = len(words_vector)
        total_words = len(dataset_words)
        self.logger.info(f'Number of cluwords {n_words}')
        return words_vector, n_words, f"Percent of vocabulary in embedding: {round(n_words / total_words, 2)}"

    @staticmethod
    def _create_cluwords(words_vector: np.ndarray):
        """Create the cluwords representation.

        Args:
            words_vector (np.ndarray): The words vector.

        Returns:
            tuple: A tuple containing the space vector and the vocabulary of cluwords.

        """
        space_vector = [np.array([round(y, 9) for y in words_vector[x].tolist()]) for x in words_vector]
        space_vector = np.array(space_vector)
        vocab_cluwords = np.asarray([x for x in words_vector])
        return space_vector, vocab_cluwords

    @staticmethod
    def _calcule_similarity(space_vector: np.ndarray, k_neighbors: int , n_threads: int):
        """Calculate the cosine similarity between cluwords.

        Args:
            space_vector (np.ndarray): The space vector of cluwords.
            k_neighbors (int): The number of nearest neighbors to consider.
            n_threads (int): The number of threads to use for computation.

        Returns:
            tuple: A tuple containing the distances and indices of the nearest neighbors.

        """
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='cosine', n_jobs=n_threads).fit(
            space_vector)
        distances, indices = nbrs.kneighbors(space_vector)
        return distances, indices

    def _compute_tf(self, n_words: int , vocab_cluwords: list, data: csr_matrix, list_cluwords: np.ndarray):
        """Compute the term frequency (tf) of cluwords.

        Args:
            n_words (int): The number of cluwords.
            vocab_cluwords (list): The vocabulary of cluwords.
            data (csr_matrix): The input data.
            list_cluwords (np.ndarray): The list of cluwords.

        Returns:
            tuple: A tuple containing the cluwords tf-idf matrix, the hyp_aux matrix, the tf matrix, and the number of cluwords.

        """
        tf_vectorizer = CountVectorizer(max_features=n_words, binary=False, vocabulary=vocab_cluwords,
                                        tokenizer=lambda x: x.split(' '))
        tf = csr_matrix(tf_vectorizer.fit_transform(data), dtype=np.float32)
        n_cluwords = len(vocab_cluwords)
        self.logger.info(f'tf shape {tf.shape}')
        hyp_aux = []
        for w in range(0, n_cluwords):
            hyp_aux.append(np.asarray(list_cluwords[w], dtype=np.float32))
        hyp_aux = np.asarray(hyp_aux, dtype=np.float32)
        hyp_aux = csr_matrix(hyp_aux, shape=hyp_aux.shape, dtype=np.float32)  # ?test sparse matrix!

        cluwords_tf_idf = tf.dot(hyp_aux.transpose())
        print(cluwords_tf_idf.shape)
        return cluwords_tf_idf, hyp_aux, tf, n_cluwords

    def _compute_idf(self, hyp_aux: csr_matrix, tf: csr_matrix, n_documents: int):
        """Compute the inverse document frequency (idf) of cluwords.

        Args:
            hyp_aux (csr_matrix): The hyp_aux matrix.
            tf (csr_matrix): The tf matrix.
            n_documents (int): The number of documents.

        Returns:
            np.ndarray: The cluwords idf.

        """
        _dot = tf.dot(hyp_aux.transpose())

        self.logger.info('Divide hyp_aux by itself')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hyp_aux_bin = csr_matrix((hyp_aux > 0) * 1., dtype=np.float32)

        self.logger.info('Dot tf and bin hyp_aux')

        _dot_bin = tf.dot(
            hyp_aux_bin.transpose())  # calcula o número de termos de uma cluword em cada documento: soma(tf * term_clu (0-1))
        _dot_bin.data = 1 / _dot_bin.data
        # n_termos_cluwords por documento

        self.logger.info('Divide _dot and _dot_bin')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mu_hyp = _dot.multiply(_dot_bin)
            print(mu_hyp.shape)
            mu_hyp = np.nan_to_num(mu_hyp, copy=False)
            # cluword / n_termos de uma cluword em um documento

        self.logger.info('Sum')
        cluwords_idf = np.sum(mu_hyp, axis=0)  # o somátorio d

        cluwords_idf[cluwords_idf == .0] = 0.001
        self.logger.info('log')
        cluwords_idf = np.log10(np.divide(n_documents, cluwords_idf))  # calcula o idf
        return cluwords_idf

    @staticmethod
    def _normalize_tfidf(X: np.ndarray, norm: str):
        """Normalize the tf-idf matrix.

        Args:
            X (np.ndarray): The tf-idf matrix.
            norm (str): The normalization method.

        Returns:
            np.ndarray: The normalized tf-idf matrix.

        """
        normalize = Normalizer(norm=norm)
        return normalize.fit_transform(X)

    @staticmethod
    def _filter_cluwords(n_words: int, threshold: float, indices: np.ndarray, distances: np.ndarray):
        """Filter the cluwords based on a similarity threshold.

        Args:
            n_words (int): The number of cluwords.
            threshold (float): The similarity threshold.
            indices (np.ndarray): The indices of the nearest neighbors.
            distances (np.ndarray): The distances to the nearest neighbors.

        Returns:
            np.ndarray: The filtered list of cluwords.

        """
        list_cluwords = np.zeros((n_words, n_words), dtype=np.float16)
        if threshold:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    # .875, .75, .625, .50
                    if 1 - distances[p][i] >= threshold:
                        list_cluwords[p][k] = round(1 - distances[p][i], 2)
                    else:
                        list_cluwords[p][k] = 0.0
        else:
            for p in range(0, n_words):
                for i, k in enumerate(indices[p]):
                    list_cluwords[p][k] = round(1 - distances[p][i], 2)
        return list_cluwords

    def generate_representation(self, embedding_file: str, embedding_binary: bool, k_neighbors: int,
                                n_threads: int, threshold: float, norm: str | None = None):
        """Generate the topic representation using CluWords algorithm.

        Args:
            embedding_file (str): The path to the word embedding file.
            embedding_binary (bool): Whether the embedding file is in binary format.
            k_neighbors (int): The number of nearest neighbors to consider.
            n_threads (int): The number of threads to use for computation.
            threshold (float): The similarity threshold for filtering cluwords.
            norm (str, optional): The normalization method for tf-idf matrix. Defaults to None.

        """
        t1 = time.time()
        self.logger.info('building embedding...')
        words_vector, n_words, vocabulary_embedding = self._build_embedding(embedding_file, embedding_binary, self.text)

        self.logger.info('creating cluwords...')
        space_vector, self.vocab = self._create_cluwords(words_vector)
        del words_vector

        self.logger.info('getting cosine similarity...')
        distances, indices = self._calcule_similarity(space_vector, k_neighbors, n_threads)

        self.logger.info('filtering cluwords...')
        list_cluwords = self._filter_cluwords(n_words, threshold, indices, distances)
        del space_vector
        del distances
        del indices

        self.logger.info('calculating cluwords tf...')
        cluwords_tf_idf, hyp_aux, tf, self.n_cluwords = self._compute_tf(n_words, self.vocab, self.text,
                                                                   list_cluwords)
        del list_cluwords

        self.logger.info('calculating cluwords idf')
        cluwords_idf = self._compute_idf(hyp_aux, tf, self.n_documents)
        del hyp_aux
        del tf
        cluwords_tf_idf = cluwords_tf_idf.multiply(cluwords_idf)
        self.matrix = csr_matrix(cluwords_tf_idf, shape=cluwords_tf_idf.shape, dtype=np.float32)

        if norm:
            self.matrix = self._normalize_tfidf(self.matrix, norm)

        self.logger.info(f'time for create cluwords: {(time.time() - t1) / 60}')

    def get_topics(self, n_topics: int = 10, n_top_words: int = 10, save_path: str = '', **kwargs):
        """Get the topics using NMF algorithm.

        Args:
            n_topics (int, optional): The number of topics to generate. Defaults to 10.
            n_top_words (int, optional): The number of top words to include in each topic. Defaults to 10.
            save_path (str, optional): The path to save the topics. Defaults to ''.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The list of topics.

        """
        t1 = time.time()
        self.logger.info('\nFitting the NMF model (kullback-leibler) with CluWords-IDF features, '
              f'n_samples={self.n_documents} and n_features={self.n_cluwords}...')

        topics = []
        nmf = NMF_sklearn(n_components=n_topics, random_state=1, init="nndsvda", beta_loss="kullback-leibler",
                  solver="mu",
                  max_iter=1000,
                  alpha_W=0.00005,
                  alpha_H=0.00005,
                  l1_ratio=0.5).fit(self.matrix)

        self.logger.info(f'time for NMF: {(time.time() - t1) / 60}')

        for topic in nmf.components_:
            topics.append(
                [self.vocab[i] for i in topic.argsort()[:-kwargs.get("n_words_clean", 101) - 1:-1]]
            )
        topics = clean_topics(topics, n_top_words)
        if save_path:
            save_topics(topics, n_top_words, save_path)
            dominant_topics(nmf, self.matrix, save_path, self.text, self.ids)
        return topics