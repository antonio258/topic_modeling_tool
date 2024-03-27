import pandas as pd


class Reader:
    def __init__(self, path: str = '', data: pd.DataFrame = None, **kwargs):
        if not path and not data:
            raise ValueError('Either path or data must be provided')

        if path:
            if not kwargs.get('text_column'):
                raise ValueError('text_column must be provided if path is provided')
            self.data = pd.read_csv(path, sep=kwargs.get('sep', ','))
            self.text = self.data[kwargs.get('text_column')]
            self.n_documents = len(self.text)
            if ids := kwargs.get('id_column'):
                self.ids = self.data[ids]
            else:
                self.ids = list(range(self.n_documents))

    def get_text(self):
        return self.text

    def get_ids(self):
        return self.ids

    def get_data_to_model(self):
        return self.text, self.ids, self.n_documents

    def __iter__(self):
        for idx in range(self.n_documents):
            yield self.text[idx]

    def __getitem__(self, idx):
        return self.text[idx]

    def __len__(self):
        return self.n_documents
