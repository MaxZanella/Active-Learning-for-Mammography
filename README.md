# Active-Learning-for-Mammography
This repository includes the Active Learning strategies assessed in the Master Thesis: Deep Learning in Mammography realized by Maxime Zanella and supervised by Professor Benoit Macq at UCLouvain.
Each method is explained with pseudo-code in the Appendices. This repository aims to give further implementations details. The model used is implemented in the file model.py.

1. MaxEntropy.py: the file contains the uncertainty measure using entropy.
2. Greedy_core_set.py: the file contains the greedy version of the core-set approach.
3. discriminative_active_learning.py: the file contains the discriminative model as well as the training procedure.
4. ensemble.py: the file contains two query methods: the one using variation ratio adn the other using MaxEntropy.
5. Monte_Carlo_dropout.py: the file contains the forward passes inference.
6. learning_loss.py, learning_loss_net.py, random_search_learning_loss.py: the files contains respectively the loss prediction performed during sampling, the learning loss module implementation, the particular random search performed for the MNIST dataset


