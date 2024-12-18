from tm_module.topic_modeling import TopicModeling
from tm_module.post_processing import save_topics, dominant_topics, clean_topics
from tm_module.utils.logger import Logger
from sklearn.cluster import KMeans
from bertopic import BERTopic as BERTopic_
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tm_module.norm_entity import norm_entity
from tm_module.post_processing import save_topics
from tm_module.utils.reader import Reader
from sklearn.cluster import AgglomerativeClustering

class BERTopic(TopicModeling):
    """BERTopic is a class that represents a topic modeling model based on BERT embeddings.

    Args:
        reader (Reader): The reader object used to read the input data.

    Attributes:
        logger (Logger): The logger object used for logging.

    Methods:
        generate_representation: Generates the representation of the input data.
        _dominant_topics: Computes the dominant topics and saves the results.
        get_topics: Fits the BERTopic model and returns the generated topics.

    """

    def __init__(self, reader: Reader):
        super().__init__(reader)
        self.logger = Logger('BERTopic').get_logger()

    def generate_representation(self):
        """Generates the representation of the input data."""
        pass

    @staticmethod
    def _dominant_topics(model: BERTopic_, data: list, path: str, ids: list, entities: list = []):
        """Computes the dominant topics and saves the results.

        Args:
            model (BERTopic_): The BERTopic model.
            data (list): The input data.
            path (str): The path to save the results.
            ids (list): The list of IDs.
            entities (list, optional): The list of entities. Defaults to [].

        """
        topicnames = ['Topico ' + str(i) for i in model.get_topic_info()["Topic"].values.tolist()]
        papernames = [str(i) for i in ids]
        topic_dist, _ = model.approximate_distribution(data)
        if "Topico -1" in topicnames:
            temp_array = 1 - topic_dist.sum(axis=1)
            topic_dist = np.insert(topic_dist, 0, temp_array, axis=1)

        df_document_topic = pd.DataFrame(np.round(topic_dist, 4), columns=topicnames)
        df_document_topic['id'] = papernames
        df_document_topic['dominant_topic'] = model.topics_

        sns.countplot(x=df_document_topic.dominant_topic)
        plt.savefig(path + "Topicos_Dominantes.png")
        plt.close()

        df_document_topic.to_csv(path + "Topicos_Dominantes.csv", sep="|")
        resumo = pd.DataFrame()
        resumo['papers'] = papernames
        resumo['dominant_topic'] = df_document_topic['dominant_topic'].values
        resumo.to_csv(path + "Resumo_Topicos_Dominantes.csv", index=False)

        df_copy = df_document_topic.copy().reset_index(drop=True)
        data = data.reset_index(drop=True)

        if entities:
            norm_entity(df_copy, data, 'id', entities, path)

    def get_topics(self, n_topics: int = 10, n_top_words: int = 10, save_path: str = '', **kwargs):
        """Fits the BERTopic model and returns the generated topics.

        Args:
            n_topics (int, optional): The number of topics to generate. Defaults to 10.
            n_top_words (int, optional): The number of top words per topic. Defaults to 10.
            save_path (str, optional): The path to save the generated topics. Defaults to ''.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The generated topics.

        """

        embedding_model = kwargs.get('embedding_model', 'all-MiniLM-L6-v2')
        if embedding_model == 'distilbert-base-cased':
            from transformers.pipelines import pipeline

            embedding_model = pipeline("feature-extraction", model="distilbert-base-cased")
        self.logger.info(f'Fitting the BERTopic model, n_samples={self.n_documents}...')
        cluster_method = kwargs.get("cluster", "hdbscan")
        if cluster_method == "hdbscan":
            model = BERTopic_(language=kwargs.get('language', 'english'), nr_topics=n_topics,
                              calculate_probabilities=True, verbose=True, top_n_words=n_top_words*2,
                              embedding_model=embedding_model)
        elif cluster_method == "kmeans":
            cluster_model = KMeans(n_clusters=n_topics)
            model = BERTopic_(language=kwargs.get('language', 'english'), calculate_probabilities=True,
                              verbose=True, top_n_words=n_top_words*2,
                              hdbscan_model=cluster_model,
                              embedding_model=embedding_model)
        elif cluster_method == "aglomerative":
            cluster_model = AgglomerativeClustering(n_clusters=n_topics)
            model = BERTopic_(language=kwargs.get('language', 'english'), calculate_probabilities=True,
                              verbose=True, top_n_words=n_top_words*2,
                              hdbscan_model=cluster_model,
                              embedding_model=embedding_model)

        model.fit(self.text)
        topics = [x.strip().split(' ')[1:][:(n_top_words*2)]
                  for x in model.generate_topic_labels(nr_words=kwargs.get("n_words_clean", 101), separator=" ")]
        topics = clean_topics(topics, n_top_words)
        if save_path:
            save_topics(topics, n_top_words, save_path)
            self._dominant_topics(model, self.text, save_path, self.ids)
        return topics
