
from ctgan.synthesizers.ctgan import CTGANSynthesizer

def fit_ctgan(data, epochs, learning_rate, embedding_dim, seed, discrete_columns):
    """
    Fits a CTGAN model to synthetic data using the provided parameters.
    
    Args:
      data: The data parameter is the input dataset that you want to use for training the CTGAN model.
    It should be a pandas DataFrame or a numpy array.
      epochs: The number of training epochs for the CTGAN model.
      learning_rate: The learning rate is a hyperparameter that determines how quickly the model learns
    from the data. It controls the step size at each iteration of the optimization algorithm. A higher
    learning rate can lead to faster convergence, but it may also cause the model to overshoot the
    optimal solution. On the other hand,
      embedding_dim: The embedding dimension is the size of the latent space representation for each
    feature in the dataset. It determines the dimensionality of the synthetic data generated by the
    CTGAN model.
      seed: The seed parameter is used to set the random seed for reproducibility. It ensures that the
    same random numbers are generated every time the code is run, which is useful for debugging and
    comparing different runs of the model.
      discrete_columns: The `discrete_columns` parameter is a list of column names or indices that
    represent discrete or categorical variables in your dataset. These variables will be treated
    differently during the training process to ensure that the generated synthetic data preserves the
    distribution of these variables.
    
    Returns:
      the trained CTGANSynthesizer model.
    """
    

    syn_model = CTGANSynthesizer(embedding_dim=embedding_dim, generator_dim=(256, 256), discriminator_dim=(256, 256),
                    generator_lr=learning_rate, generator_decay=1e-6, discriminator_lr=learning_rate,
                    discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                    log_frequency=True, verbose=False, epochs=epochs, pac=10, cuda=True)

    syn_model.set_random_state(seed)
    syn_model.fit(train_data=data, discrete_columns=discrete_columns)

    return syn_model 