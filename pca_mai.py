import numpy as np

def my_pca(data, print_details = False):
    try:
        data = data.to_numpy()
    except:
        pass
    data = data.T
    cov_matrix = np.cov(data)
    if print_details:
        print("cov matrix:")
        print(cov_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    if print_details:
        print("eigen vectors:")
        print(eigenvectors)
        print("eigen values:")
        print(eigenvalues)
    idxs = np.flip(np.argsort(eigenvalues))
    sorted_vectors = eigenvectors[:, idxs]
    sorted_values = eigenvalues[idxs]
    if print_details:
        print("idx sort array:")
        print(idxs)
    data = data.T
    changed_bases = []
    transform_bases = np.linalg.inv(sorted_vectors)
    for vector in data:
        changed_bases.append(transform_bases.dot(vector))
    changed_bases = np.array(changed_bases)
    return changed_bases