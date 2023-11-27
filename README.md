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

#### Model Architecture

The Transformer model is defined in the `TransformerModel` class. Key parameters include:
- Number of tokens (ntoken)
- Input dimension (ninp)
- Number of heads in multihead attention (nhead)
- Dimension of feedforward network (nhid)
- Number of transformer layers (nlayers)
- Dropout rate (dropout)

The model includes positional encoding and uses PyTorch's nn.Transformer module for the core transformer architecture.

#### Training and Evaluation

The model is trained using cross-entropy loss and optimized with Adam. Training involves multiple epochs over the prepared datasets. The model's performance is evaluated on a separate test dataset.

### Datasets and DataLoaders

Custom datasets and data loaders are created for both training and test sets, enabling efficient batch processing during model training.

### Memory Management

Garbage collection and CUDA cache clearing are employed to manage memory, especially when using GPU for training.
