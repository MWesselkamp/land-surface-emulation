import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names='', save_to=''):
    # Get feature importances from the model
    importances = model.feature_importances_

    # Sort indices in descending order
    indices = importances.argsort()[::-1]
    # Rearrange feature names based on sorted indices
    sorted_feature_names = [feature_names[i] for i in indices]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), sorted_feature_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_to)

def plot_increments(preds, y_test, save_to=''):

    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    axs = axs.flatten()
    for i in range(preds.shape[1]):
        axs[i].plot(preds[:,i], label="Predictions")
        axs[i].plot(y_test[:,i], label="True values")
        axs[i].legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_to)


