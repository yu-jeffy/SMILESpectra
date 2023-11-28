# SMILESpectra
Transformer-based deep learning model to convert SMILES chemical compounds into mass spectra, trained on a ZINC12 dataset with in silico generated mass spectra from FragGenie, and validated with real-world examples from MassBank.

## Data Sources

### MassBank
MassBank data can be accessed at:
[MassBank](https://mona.fiehnlab.ucdavis.edu/downloads)

### ZINC
For ZINC dataset, visit:
[ZINC Standard Subsets](https://zinc12.docking.org/browse/subsets/standard)

## Encoder Tool

### SmilesPE
The SmilesPE encoder is available on GitHub:
[SmilesPE GitHub Repository](https://github.com/XinhaoLi74/SmilesPE)



## Transformer Model

### Model Architecture

The `TransformerModel` in the code is a PyTorch-based architecture designed for chemical data processing. Its structure includes an embedding layer (`nn.Embedding`) with a vocabulary size of `ntoken=3002` and an input dimension of `ninp=1024`. The model utilizes a transformer core (`nn.Transformer`) configured with `nhead=16` heads in its multi-head attention mechanism, `nhid=4096` dimensions in its feedforward network, and `nlayers=12` encoder and decoder layers. A dropout rate of `dropout=0.2` is applied for regularization. Positional encoding is integrated to maintain sequence order, essential for transformer models. The output is mapped back to token space through a linear layer (`nn.Linear`).

### Training and Evaluation

The model is trained using cross-entropy loss and optimized with Adam. Training involves multiple epochs over the prepared datasets. The model's performance is evaluated on a separate test dataset.

### Datasets and DataLoaders

Custom datasets and data loaders are created for both training and test sets, which batch the data into smaller sizes for processing.
### Memory Management

Garbage collection and CUDA cache clearing are employed to manage memory, especially when using GPU for training.
