import numpy as np
from QMC_sampler import QMC_sampler

# Assuming QMC_sampler class is already defined above

def test_qmc_sampler():
    # Create a QMC_sampler instance
    qmc_sampler = QMC_sampler()
    
    # Test initialize_from_file function (replace with an actual path to your file)
    try:
        qmc_sampler.initialize_from_file('genvec.txt')
        print("QMC Sampler initialized successfully from file.")
        print("Generating vector:", qmc_sampler.generating_vector)
        print("Max level:", qmc_sampler.max_level)

    except Exception as e:
        print(f"Error initializing QMC sampler from file: {e}")
    
    # Add levels and print the generated vectors and matrices
    num_levels_to_add = 5  # Number of levels to add
    
    for level in range(1, num_levels_to_add + 1):
        print(f"\n--- Adding Level {level} ---")
        qmc_sampler.add_level()
        print(f"Max level: {qmc_sampler.max_level}")
        
        # Print the most recent vector generated at this level
        print(f"Generated vector at level {level}: {qmc_sampler.gen_vecs[-1]}")
        
        # Print the matrix of points generated at this level
        print(f"Generated matrix of points at level {level}:")
        print(qmc_sampler.points[-1])
    
    # Print the final status of the QMC sampler
    print("\nFinal Status of QMC Sampler:")
    print(f"Max level: {qmc_sampler.max_level}")
    print(f"Number of vectors generated: {len(qmc_sampler.gen_vecs)}")
    print(f"Number of matrices of points generated: {len(qmc_sampler.points)}")
    
    # Optionally, print the last generated matrix
    print(f"Last generated matrix of points:")
    print(qmc_sampler.points[-1])

# Run the test
if __name__ == "__main__":
    test_qmc_sampler()
