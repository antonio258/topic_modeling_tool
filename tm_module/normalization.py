import pandas as pd
import numpy as np
import os
import ast

def normaliza_valores_internos(vetor):
    """Normalize the internal values of a vector.

    Args:
        vetor (list): The input vector.

    Returns:
        list: The normalized vector.
    """
    entrou = False
    for v in range(len(vetor)):
        if(vetor[v] > 0.0):
            entrou  = True
            break

    if(entrou == False):
        vetor[0] = 1.0

    vetor = np.array(vetor)
    
    return vetor/sum(vetor) * 100

def normaliza(vetor):
    """Normalize a vector.

    Args:
        vetor (list): The input vector.

    Returns:
        list: The normalized vector.
    """
    vetor = np.array(vetor)
    if(sum(vetor) == 0):
        vetor_ = vetor
    else:
        vetor_ = (vetor/sum(vetor)) * 100 
    
    return vetor_.tolist()

def normalize(lattes_link, data_pp, results_path, topicos_dominantes_file):
    """Normalize the topic values for each paper.

    Args:
        lattes_link (str): The file path of the input file containing paper information.
        data_pp (str): The file path of the input file containing data year information.
        results_path (str): The directory path to save the results.
        topicos_dominantes_file (str): The file name of the input file containing dominant topics.

    Returns:
        None
    """
    df = pd.read_csv(lattes_link, sep="|", dtype=str)
    data_year = pd.read_csv(data_pp, sep="|", dtype=str)
    df = df.merge(data_year[['id']], left_on='paper_id', right_on='id')

    print("Recuperando o arquivo ...")
    df_dominante = pd.read_csv('{}/{}'.format(results_path, topicos_dominantes_file), sep = "|", dtype=str)
    qtd_topicos = df_dominante.shape[1] - 2
    
    papers_id = list(df['id'])
    
    id_paper_ = []
    id_author_ = []
    id_institution_ = []
    id_uf_ = []
    id_state_ = []
    topicos_ = []
    
    df_join = pd.merge(df, df_dominante,how='inner', left_on=['id'], right_on=['Unnamed: 0'])
    fim = len(list(df_join['id']))
    contador = 0

    print('running in dataframe')
    for index, row in df_join.iterrows():
        contador += 1 
        valores_normalizados = [0 for k in range(qtd_topicos)]
        valores_positivos = []

        for q in range(qtd_topicos):
            topico_txt = "Topico " + str(q)
            valor_ = float(row[str(topico_txt)])
            valores_normalizados[q] = valor_

        valores_normalizados = normaliza_valores_internos(valores_normalizados)
        valores_normalizados = valores_normalizados.tolist()

        for q in range(qtd_topicos):
            topico_txt = "Topico " + str(q)
            if(valores_normalizados[q] > 0.0):
                valores_positivos.append({str(topico_txt): valores_normalizados[q]})
                
        topicos_.append(valores_positivos)
        id_paper_.append(row['id'])
        id_author_.append(row['author_id'])
        id_institution_.append(row['work_institution']) 
        id_uf_.append(row['uf_id'])
    
    dataframe_final = pd.DataFrame()
    dataframe_final['id'] = id_paper_
    dataframe_final['author_id'] = id_author_
    dataframe_final['work_institution'] = id_institution_
    dataframe_final['uf_id'] = id_uf_
    dataframe_final['topics'] = topicos_

    print("Salvando o arquivo ...") 
    dataframe_final.to_csv('{}/partiatable_{}'.format(results_path, topicos_dominantes_file), sep="|")

    campo_id = [ 'author_id', 'work_institution', 'uf_id']
    nome_id =  [ 'authors', 'institutions', 'uf']

    for cid, nid in zip(campo_id, nome_id):
        print("Calculando Porcentagens ...")
        fields=[cid, "topics"]
        df = pd.read_csv('{}/partiatable_{}'.format(results_path, topicos_dominantes_file), sep="|", usecols=fields, dtype=str)
        
        df = (df.set_index([cid, df.groupby([cid]).cumcount()]).unstack().sort_index(axis=1, level=1))
        df.columns = ['{}_{}'.format(i, j) for i, j in df.columns]
        df = df.reset_index()

        n_linhas = len(list(df[cid]))
        colunas = len(list(df.columns)) - 1

        novas_entidades_id = []
        topicos_normalizados_linha = []
        topicos_normalizados_coluna = []
        fim = len(list(df[cid]))
        inicio = 0

        somatorio_topico = [0 for i in range(qtd_topicos)]

        for index, row in df.iterrows():
            inicio = inicio + 1
            valores_topicos  = [0 for k in range(qtd_topicos)]
            
            for i in range(colunas):
                if(str(row['topics_' + str(i)]) != "nan"):
                    lista_topicos = ast.literal_eval(str(row['topics_' + str(i)]))
                    for l in lista_topicos:
                        for n in range(qtd_topicos):
                            try:
                                valores_topicos[n] = valores_topicos[n] + l['Topico ' + str(n)] 
                                break
                            except:
                                pass
                else:
                    break

            topicos_normalizados_linha.append(normaliza(valores_topicos))

            for v in range(len(valores_topicos)):
                somatorio_topico[v] = somatorio_topico[v] + valores_topicos[v]

            novas_entidades_id.append(row[cid])
            
    # Additional code removed for brevity
            # de acordo com o topico global
            topicos_normalizados_coluna.append(valores_topicos)

        # segunda etapa é normalizar pelo somatorio da coluna
        for t in range(len(topicos_normalizados_coluna)):
            pos = 0
            for s in somatorio_topico:
                topicos_normalizados_coluna[t][pos] = (topicos_normalizados_coluna[t][pos] / s) * 100
                pos = pos + 1

        # vou guardar os valores de todas as colunas
        temp_colunas = [[] for k in range(qtd_topicos)]
        
        for topic in topicos_normalizados_coluna:
            # pra cada linha, que tem o campo de normalizado por coluna, eu vou recuperar o valor que eu tenho
            for n in range(qtd_topicos):
                # eu estou aqui gerando um conjunto de vetores com todos os valores, em cada vetor, de um mesmo topico
                temp_colunas[n].append(topic[n])

        # terceira etapa, vamos fazer a normalizacao pelo maximo de uma coluna entre 0 e 100
        novas_colunas = []
        #print(temp_colunas)

        for coluna in temp_colunas:
            melhores_valores = []
            melhores_entidade = []
            for c in range(len(coluna)):
                if(len(melhores_valores) < 10):
                    melhores_valores.append(coluna[c])
                    melhores_entidade.append(c)
                else:
                    for mv in range(len(melhores_valores)):
                        if (coluna[c] > melhores_valores[mv]):
                            melhores_valores[mv] = coluna[c]
                            melhores_entidade[mv] = c
                            break
            # 10 melhores
            soma_melhores = sum(melhores_valores)
            # vou zerar os caras que nao estao entre os melhores
            for c in range(len(coluna)):
                entrou = False
                for me in range(len(melhores_entidade)):
                    # achei um cara que é um dos 10 - vou normalizar
                    if( c == melhores_entidade[me] ):
                        coluna[c] = (melhores_valores[me] / soma_melhores) * 100
                        entrou = True
                        break
                # zero os caras que nao sao um dos melhores
                if(entrou == False):
                    coluna[c] = 0

            # maximo = max(coluna)
            # # print("Coluna Max:" + str(maximo))
            #
            # arr = np.array(coluna)
            #
            # arr = (arr / maximo) * 100

            novas_colunas.append(coluna)

        # agora eu vou gerar novamente o vetor de topicos normalizados por coluna para um mesmo autor, instituição ou país
        topicos_normalizados_coluna = []
        cont = 0
        # vou fazer isso, percorrendo o conjunto de vetores e pegando posicao a posicao e guardando na linha do autor, instituicao ou pais
        for i in range(n_linhas):
            cont = cont + 1
            lista = [0 for k in range(qtd_topicos)]
            for nc in range(len(novas_colunas)):
                lista[nc] = novas_colunas[nc][i]  

            topicos_normalizados_coluna.append(lista)   
        
        df = pd.DataFrame()
        df[cid] = novas_entidades_id
        df['topics_normalized_row'] = topicos_normalizados_linha
        df['topics_normalized_column'] = topicos_normalizados_coluna
        df.to_csv("{}/normalized_topics_{}.csv".format(results_path, nid), sep="|")

