import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyjarowinkler import distance
from tm_module.norm_entity import norm_entity


def dominant_topics(model, matriz_tfidf, path, data, ids, entities=[]):
    """Generate dominant topics based on a topic model.

    Args:
        model (object): The trained topic model.
        matriz_tfidf (numpy.ndarray): The TF-IDF matrix of the documents.
        path (str): The path to save the generated files.
        data (pandas.DataFrame): The original data.
        ids (list): The list of document IDs.
        entities (list, optional): The list of entities to normalize. Defaults to [].

    Returns:
        None
    """
    topicnames = ['Topico ' + str(i) for i in range(model.n_components)]
    papernames = [str(i) for i in ids]
    print(matriz_tfidf.shape, len(papernames))
    df_document_topic = pd.DataFrame(np.round(model.transform(matriz_tfidf), 4), columns=topicnames)
    df_document_topic['id'] = papernames
    df_document_topic['dominant_topic'] = np.argmax(df_document_topic.drop('id', axis=1).values, axis=1)

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


def save_topics(topics, n_top_words, results_path, vocabulary_embedding=''):
    """Save the generated topics to a file.

    Args:
        topics (list): The list of topics.
        n_top_words (int): The number of top words to include in each topic.
        results_path (str): The path to save the result file.
        vocabulary_embedding (str, optional): The vocabulary embedding information. Defaults to ''.

    Returns:
        None
    """
    os.makedirs(results_path, exist_ok=True)

    n_topics = len(topics)
    with open('{}/result_topic_{}.txt'.format(results_path, n_top_words), 'w') as f_res:
        f_res.write('Topics {} N_Words {}{}\n'.format(n_topics, n_top_words, vocabulary_embedding))
        f_res.write('Topics:\n')
        topics_t = []
        for i, topic in enumerate(topics):
            topics_t.append(topic[:n_top_words])
            f_res.write('{} - '.format(i))
            for word in topic[:n_top_words]:
                f_res.write('{} '.format(word))

            f_res.write('\n')
        f_res.close()


def clean_topics(topics: list, n_top_words: int):
    """Clean the topics by removing duplicate and short words.

    Args:
        topics (list): The list of topics.
        n_top_words (int): The number of top words to keep for each topic.

    Returns:
        list: The cleaned topics.

    """
    topics_out = []
    for topic in topics:
        topic_t = [x for x in topic if x and len(x) > 1]
        topics_out.append(topic_t)

    for index, topic in enumerate(topics_out):
        filtered_topic = []
        insert_word = np.ones(len(topic))
        for w_i in range(0, len(topic) - 1):
            if insert_word[w_i]:
                filtered_topic.append(topic[w_i])
                for w_j in range((w_i + 1), len(topic)):
                    if distance.get_jaro_distance(topic[w_i], topic[w_j], winkler=True, scaling=0.1) > 0.75:
                        insert_word[w_j] = 0
        topics_out[index] = filtered_topic[:n_top_words]
    return topics_out