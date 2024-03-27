# Topic Modeling Tool

This project is a collection of Python scripts for topic modeling. The project include a variety of topic modeling techniques, such as Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), CluWords and BERTopic. The project also includes a pre-processing module for text data.

## Features

- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- CluWords
- BERTopic

## Installation

This project requires Python and pip installed. Clone the project and install the dependencies:

```bash
pip install git+https://github.com/antonio258/topic_modeling_tool.git
```

## Usage
Import the module and use the desired topic modeling technique:

```python

from tm_module.utils.reader import Reader
from tm_module.cluwords import CluWords

# Read the data
reader = Reader(path="path/to/data", id_column="id", text_column="text")

# Create the CluWords object
cluwords = CluWords(reader)
cluwords.generate_representation(
    embedding_file="path/to/embedding.vec",
    embedding_binary=False,
    k_neighbors=500,
    n_threads=20,
    threshold=0.4
)
cluwords.get_topics(10, 10, save_path="path/to/save")

```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.