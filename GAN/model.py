from sdv.single_table import CTGANSynthesizer

def create_ctgan_model(epochs=100, batch_size=500):
    """
    Create and return a CTGAN synthesizer instance with specified hyperparameters.
    """
    synthesizer = CTGANSynthesizer(epochs=epochs, batch_size=batch_size)
    return synthesizer