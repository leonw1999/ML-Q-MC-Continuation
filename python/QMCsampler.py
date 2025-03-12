import numpy as np

class QMC_sampler:
    def __init__(self):
        self.points = []  # List of vectors (initially empty)
        self.max_level = 0  # Positive integer
        self.generating_vector = None  # Stores the first vector
        self.gen_vecs = []  # List to store generated vectors
    
    def initialize_from_file(self, filename):
        """
        Reads a file with two columns: dimension index and vector component,
        constructs the first vector, and stores it in generating_vector.
        Leaves max_level at 0.
        """
        data = np.loadtxt(filename)
        
        if data.size == 0:
            raise ValueError("File is empty or not formatted correctly.")
        
        # Ensure data is at least 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Sort data by dimension index (first column)
        sorted_data = data[data[:, 0].argsort()]
        
        # Extract the vector components (second column)
        self.generating_vector = sorted_data[:, 1]
        
        # Keep the list empty and leave max_level at 0
        self.max_level = 0
    
    def add_level(self):
        """
        If max_level is 0, creates a vector with the first three entries
        of generating_vector and inserts it into points and gen_vecs.
        If max_level is not 0, creates a new vector by alternating between
        a subvector of generating_vector and the last vector in gen_vecs.
        Increases max_level by 1 at the end of the if block and at the beginning of the else block.
        If max_level is 0, adds the last vector of gen_vecs to the list points.
        If max_level is not 0, computes a 2^max_level x (2 + 2^max_level) matrix
        where the i-th row is (i * v / 2^max_level) mod 1, where v is the last vector in gen_vecs.
        Appends this matrix to points.
        """
        if self.max_level == 0 and self.generating_vector is not None and len(self.generating_vector) >= 3:
            new_vector = self.generating_vector[:3]
            self.points.append(new_vector)
            self.gen_vecs.append(new_vector)
            self.max_level += 1
        else:
            size = 2 ** self.max_level
            dim = 2 + size
            
            # First two components remain the same
            new_vector = self.generating_vector[:2].tolist()
            
            # Define subvector from generating_vector
            start_idx = 2 + 2 ** (self.max_level - 1)
            end_idx = 2 + 2 ** self.max_level
            subvector = self.generating_vector[start_idx:end_idx]
            
            # Get the last vector in gen_vecs
            last_vector = self.gen_vecs[-1][2:]
            
            # Mix the subvector and last_vector alternatingly using slicing
            mixed_vector = np.empty(len(subvector) + len(last_vector))
            mixed_vector[0::2] = subvector
            mixed_vector[1::2] = last_vector

            
            # Concatenate to form the new vector
            new_vector.extend(mixed_vector.tolist())
            
            self.gen_vecs.append(np.array(new_vector))

            # Generate lattice points
            size = 2 ** self.max_level
            v = self.gen_vecs[-1]
            indices = np.arange(size).reshape(-1, 1)
            matrix = (indices * v / size) % 1
            self.points.append(matrix)

            self.max_level += 1
    