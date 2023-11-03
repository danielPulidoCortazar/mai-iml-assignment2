def plot_pca_and_isomap(X, labels):
    # Perform PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Perform Isomap for dimensionality reduction
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(X)

    # Create subplots for PCA and Isomap visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PCA visualization
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
    axes[0].set_title('PCA Visualization')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # Plot Isomap visualization
    axes[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=labels, cmap='viridis', s=20)
    axes[1].set_title('Isomap Visualization')
    axes[1].set_xlabel('Isomap Component 1')
    axes[1].set_ylabel('Isomap Component 2')

    plt.show()
