'''
TODO: Implement here a script to run multiple instances of training for fitting thresholds
1. Understand what parameters of PSO can be changed to achieve better results:
    ->  swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)

2. Analyze the classes' distribution in the dataset:
    -> dataset_analyzer.py

3. Make a reduced dataset with hand-select images that are well-distributed on the field

4. Perform trainings with randomized annotated line segments using different sizes of training data
    and splitting them into: train (70%), validation (15%), and test (15%)

Concurrently, more data needs to be annotated.
'''