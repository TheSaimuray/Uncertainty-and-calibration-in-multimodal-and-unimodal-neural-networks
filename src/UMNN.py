# Import necessary libraries
import torch
torch.manual_seed(0)  # Set random seed for reproducibility
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
np.random.seed(0)  # Set random seed for NumPy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import uncertainty_metrics.numpy as uncertainty_metrics
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import balanced_accuracy_score

# Set the device for PyTorch (GPU if available, otherwise CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Define the unimodal neural network model with dropout and batch normalization
class NN(nn.Module):
    def __init__(self, input_size, layer_size, dropout_p):
        super().__init__()

        self.NN_layout = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.BatchNorm1d(layer_size),
            nn.Tanh(),

            nn.Linear(layer_size, layer_size//2),
            nn.BatchNorm1d(layer_size//2),
            nn.Tanh(),
            nn.Dropout(dropout_p),

            nn.Linear(layer_size//2, layer_size//2),
            nn.BatchNorm1d(layer_size//2),
            nn.Tanh(),
            nn.Dropout(dropout_p),

            nn.Linear(layer_size//2, 1)  # Output layer
        )

    def forward(self, x):
        logits = self.NN_layout(x)
        return logits
    
    def set_dropout_p(self, dropout_p):
        """Dynamically update dropout probability."""
        for module in self.NN_layout:
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

# Enable dropout during inference for Monte Carlo Dropout
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# Preprocess data: feature selection, scaling, and conversion to PyTorch tensors
def preprocessing_PL(X_train, y_train, X_test, y_test, input_size):

    # Select top features using chi-square test (initial filtering)
    chi2_selector = SelectKBest(score_func=chi2, k=input_size*2)
    X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
    X_test_chi2 = chi2_selector.transform(X_test)

    # Apply mRMR feature selection to further reduce dimensionality
    selected_features = mrmr_classif(X = pd.DataFrame(X_train_chi2), y = pd.DataFrame(y_train), K = input_size, show_progress=False, n_jobs=-1)
    X_train_sel = X_train_chi2[:, selected_features]
    X_test_sel = X_test_chi2[:, selected_features]

    # Normalize the selected features using StandardScaler
    scaler = StandardScaler()
    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)

    # Convert data to PyTorch tensors and move to the appropriate device
    X_train_T = torch.tensor(X_train_sel, dtype=torch.float32)
    y_train_T = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    
    X_test_T = torch.tensor(X_test_sel, dtype=torch.float32)
    y_test_T = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_T.to(device), y_train_T.to(device), X_test_T.to(device), y_test_T.to(device)

# Train the model and evaluate on the validation set
def train_model(model: NN, optimizer, loss_fn, X_train_T, y_train_T, X_val_T, y_val_T, hyperparameters, plot_info:list): 
    train_losses = []
    val_accuracies = []

    for epoch in range(hyperparameters['epochs']):
        model.train()
        epoch_loss = 0

        # Mini-batch training
        for batch in range(0, len(X_train_T), hyperparameters['batch_size']):
            X_train_T_batch = X_train_T[batch: batch+hyperparameters['batch_size']]
            y_train_T_batch = y_train_T[batch: batch+hyperparameters['batch_size']]

            if X_train_T_batch.size(0) > 1:
                optimizer.zero_grad()
                outputs = model(X_train_T_batch)
                loss = loss_fn(outputs, y_train_T_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Accumulate loss per batch

        if plot_info is not None:
            avg_epoch_loss = epoch_loss / (len(X_train_T) / hyperparameters['batch_size'])
            train_losses.append(avg_epoch_loss)  
            temp_val_acc = eval_model(model, X_val_T, y_val_T)
            val_accuracies.append(temp_val_acc)

    # Plot loss and accuracy over epochs
    if plot_info is not None:
        plot_loss_balAcc(train_losses, val_accuracies, hyperparameters, plot_info)
        
    return model

# Evaluate model performance using balanced accuracy
def eval_model(model: NN, X_test_T, y_test_T):
    model.eval()
    with torch.no_grad():
        y_pred_test = torch.sigmoid(model(X_test_T))  # Apply sigmoid for binary classification
        y_pred_test = (y_pred_test > 0.5).float()  # Convert to binary predictions
        test_bal_acc = balanced_accuracy_score(y_test_T.cpu().numpy(), y_pred_test.cpu().numpy())
    return test_bal_acc

# Plot training loss and balanced accuracy over epochs
def plot_loss_balAcc(train_losses, val_accuracies, hyperparameters, plot_info):
    epochs_range = range(1, hyperparameters['epochs'] + 1)
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, marker='o', color='orange', label="Balanced accuracy")
    plt.xlabel("Epochs")
    plt.ylabel('Balanced accuracy')
    plt.title("Balanced Accuracy Over Epochs")
    plt.legend()

    # Save the plot
    save_path = f'...'
    plt.savefig(f'{save_path}/{plot_info[0]}_UM_{plot_info[1]}_{plot_info[2]}_{plot_info[3]}_{plot_info[4]}.png')
    plt.close()

# Generate hyperparameters using random sampling
def hyperparameters(np_seed):
    np.random.seed(np_seed)

    return dict(
        lr = 10 ** np.random.choice(np.arange(-6, -2, 0.5)),
        wd = 10 ** np.random.choice(np.arange(-5, -1, 0.5)),
        dropout_p = np.random.choice(np.arange(0.1, 0.8, 0.05)),
        epochs = np.random.choice(np.arange(5, 201, 5)),
        batch_size = 2 ** np.random.randint(3, 6),
        input_size = np.random.choice(np.arange(50, 5001, 50)),
        layer_size = np.random.choice(np.arange(20, 5001, 10)),
        m_input_size = np.random.choice(np.arange(50, 532, 5)),
        m_layer_size = np.random.choice(np.arange(20, 501, 10))
    )

# Perform Monte Carlo Dropout for uncertainty estimation
def monte_carlo_dropout(model, X_test_T, forward_passes):
    model.eval()
    enable_dropout(model)  # Enable dropout at test time
    predictions = []

    for _ in range(forward_passes):
        with torch.no_grad():
            output = torch.sigmoid(model(X_test_T))
            predictions.append(output.unsqueeze(0))

    predictions_T = torch.cat(predictions, dim=0)
    mean_predictions_T = predictions_T.mean(dim=0)
    variance_T = predictions_T.var(dim=0)
    mean_variance = variance_T.mean()
    
    return mean_predictions_T, mean_variance, variance_T, predictions_T

def calculate_uncertainty_ratio(mean_predictions_T, variance_T, y_test_T):
    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    binary_predictions = (mean_predictions_T > 0.5).float()
    
    # Initialize masks to track correct and incorrect predictions
    correct_mask = torch.zeros_like(binary_predictions)
    incorrect_mask = torch.zeros_like(binary_predictions)
    
    # Identify correct and incorrect predictions by comparing with ground truth labels
    for i in range(len(binary_predictions)):
        if binary_predictions[i] == y_test_T[i]:
            correct_mask[i] = 1.0  # Mark as correct
            incorrect_mask[i] = 0.0
        else:
            correct_mask[i] = 0.0
            incorrect_mask[i] = 1.0  # Mark as incorrect
    
    # Calculate average variance for correct and incorrect predictions
    correct_variance = (variance_T * correct_mask).sum() / correct_mask.sum()
    incorrect_variance = (variance_T * incorrect_mask).sum() / incorrect_mask.sum()
    
    # Compute uncertainty ratio as the ratio of incorrect variance to correct variance
    uncertainty_ratio = incorrect_variance / correct_variance
    
    return uncertainty_ratio

def calculate_entropy(mean_predictions_T):
    # Clamp probabilities to avoid log(0), which would result in NaN values
    mean_predictions_T = torch.clamp(mean_predictions_T, min=1e-8, max=1 - 1e-8)

    # Compute entropy for each sample using the standard entropy formula
    entropy = -mean_predictions_T * torch.log(mean_predictions_T) - (1 - mean_predictions_T) * torch.log(1 - mean_predictions_T)

    return entropy  # Return entropy values for all samples

# ###########################################################

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['cancer_type', 'data_type', 'Ball_acc', '0.2_mean_variance', '0.2_uncertainty_ratio', '0.2_mean_entropy',
                                   '0.5_mean_variance', '0.5_uncertainty_ratio', '0.5_mean_entropy',
                                   '0.8_mean_variance', '0.8_uncertainty_ratio', '0.8_mean_entropy', '8_ACE', '16_ACE'])

# Initialize a list to temporarily store results before appending to DataFrame
res_list = []

# Define the cancer types to analyze
which_cancer = ['OV', 'BRCA']
for cancer_type in which_cancer:
    res_list.append(cancer_type)

    # Define the data types (modalities) to analyze
    dataTypes = ['RNA', 'miRNA']

    for data_type in dataTypes:
        res_list.append(data_type)
        
        # Define file path for loading data
        path = f'...'

        # Load the data
        data = pd.read_csv(path, sep='\t')
        
        # Separate features (X) and labels (y)
        X = data.drop(columns='class').values
        y = data['class'].values

        # Define the number of hyperparameter tuning iterations
        n_iter = 60

        # Lists to store test performance metrics
        test_bal_acc_list = []  # Balanced accuracy
        test_cm_list = []  # Confusion matrix

        # Outer cross-validation setup
        outer_loop_index = 1
        outer_splits = 4
        inner_splits = 3
        np_seed = 42  # Set seed for reproducibility

        outer_cv = StratifiedKFold(n_splits=outer_splits)
        for train_val_idx, test_idx in outer_cv.split(X, y):
        
            # Split data into training-validation and test sets
            X_train_val = X[train_val_idx]
            y_train_val = y[train_val_idx]

            X_test = X[test_idx]
            y_test = y[test_idx]

            # Track the best validation balanced accuracy and corresponding hyperparameters
            mean_best_val_bal_acc = 0
            best_val_par = None
            inner_loop_index = 1

            # Hyperparameter tuning loop
            for k in range(n_iter):
                hyperparameters_list = hyperparameters(np_seed)
                np_seed += 1  # Update seed for randomness
                temp_val_bal_acc = []  # Store balanced accuracy per iteration
                
                # Adjust input size for miRNA data
                if data_type == 'miRNA':
                    hyperparameters_list['input_size'] = hyperparameters_list['m_input_size']

                # Inner cross-validation setup
                inner_cv = StratifiedKFold(n_splits=inner_splits)
                for train_idx, val_idx in inner_cv.split(X_train_val, y_train_val):
                    
                    # Split into training and validation sets
                    X_train = X_train_val[train_idx]
                    y_train = y_train_val[train_idx]

                    X_val = X_train_val[val_idx]
                    y_val = y_train_val[val_idx]

                    # Preprocess data for PyTorch model
                    (X_train_T, y_train_T, X_val_T, y_val_T) = preprocessing_PL(X_train, y_train, X_val, y_val, hyperparameters_list['input_size'])

                    # Initialize model with current hyperparameters
                    model = NN(hyperparameters_list['input_size'], hyperparameters_list['layer_size'], hyperparameters_list['dropout_p']).to(device)
                        
                    loss_fn = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters_list['lr'], weight_decay=hyperparameters_list['wd'])

                    # Train the model
                    inner_trained_model = train_model(model, optimizer, loss_fn, X_train_T, y_train_T, X_val_T, y_val_T, hyperparameters_list,
                                                      plot_info=['VAL', cancer_type, data_type, outer_loop_index, inner_loop_index])
                    
                    # Evaluate model on validation data
                    bal_acc = eval_model(inner_trained_model, X_val_T, y_val_T)
                    temp_val_bal_acc.append(bal_acc)
                    inner_loop_index += 1

                # Compute mean and standard deviation of balanced accuracy for current hyperparameter set
                mean_temp_val_bal_acc = sum(temp_val_bal_acc) / len(temp_val_bal_acc)

                # Update best validation accuracy and hyperparameters if necessary
                if mean_temp_val_bal_acc > mean_best_val_bal_acc:
                    mean_best_val_bal_acc = mean_temp_val_bal_acc
                    best_val_par = hyperparameters_list
            
            # Preprocess final train and test sets
            (X_train_val_T, y_train_val_T, X_test_T, y_test_T) = preprocessing_PL(X_train_val, y_train_val, X_test, y_test, best_val_par['input_size'])

            # Train final model using best hyperparameters
            model_evaluation_test = NN(best_val_par['input_size'], best_val_par['layer_size'], best_val_par['dropout_p']).to(device)
            optimizer_test = optim.Adam(model_evaluation_test.parameters(), lr=best_val_par['lr'], weight_decay=best_val_par['wd'])
            loss_fn_test = nn.BCEWithLogitsLoss()

            trained_model_test = train_model(model_evaluation_test, optimizer_test, loss_fn_test, X_train_val_T, y_train_val_T, X_test_T, y_test_T, best_val_par,
                                             plot_info=['TEST', cancer_type, data_type, outer_loop_index, ''])

            # Evaluate final model on test set
            test_bal_acc = eval_model(trained_model_test, X_test_T, y_test_T)
            test_bal_acc_list.append(test_bal_acc)
            res_list.append(float(test_bal_acc))

            # Monte Carlo Dropout for uncertainty estimation
            drop_p = [0.2, 0.5, 0.8]
            foreward_passes = 50

            # Iterate through different dropout probabilities
            for p in drop_p:

                # Set the dropout probability for the trained model
                trained_model_test.set_dropout_p(p)

                # Perform Monte Carlo Dropout to obtain predictions and uncertainty estimates
                mean_predictions_T, mean_variance, variance_T, predictions_T = monte_carlo_dropout(trained_model_test, X_test_T, foreward_passes)

                # Compute uncertainty ratio using mean predictions and variance
                uncertainty_ratio = calculate_uncertainty_ratio(mean_predictions_T, variance_T, y_test_T)

                # Compute entropy for uncertainty measurement
                entropy = calculate_entropy(mean_predictions_T)

                # Compute the mean entropy across samples
                mean_entropy = entropy.mean()

                # Append the computed uncertainty metrics to the results list
                res_list.append(float(mean_variance))
                res_list.append(float(uncertainty_ratio))
                res_list.append(float(mean_entropy))

                # Convert tensor predictions to numpy array for visualization
                mean_predictions_for_plot = mean_predictions_T.cpu().detach().numpy().flatten()

                # Plot a histogram of predicted confidences
                plt.hist(mean_predictions_for_plot, bins=10, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('Confidence')
                plt.ylabel('Frequency')
                plt.title('Histogram of Predicted Confidences')

                # Save the histogram plot to the specified path
                plt.savefig(f'/mimer/NOBACKUP/groups/naiss2023-22-417/simone_results/plots/confidence_plots/UM/UM_{cancer_type}_{data_type}_{outer_loop_index}_Confidence_plot_p{p}.png')
                plt.close('all')

            # Set the model to evaluation mode
            trained_model_test.eval()

            # Disable gradient computation for efficiency
            with torch.no_grad():
                # Obtain model predictions with sigmoid activation
                um_predictions_T = torch.sigmoid(trained_model_test(X_test_T))

            # Convert true labels to a numpy array
            true_lables_um = y_test_T.cpu().detach().numpy().flatten().astype(int)

            # Construct an array with class 0 and class 1 probabilities
            predictions_um = np.hstack([
                1 - um_predictions_T.cpu().detach().numpy(),  # Probability of class 0
                um_predictions_T.cpu().detach().numpy()       # Probability of class 1
            ])

            # Compute Expected Calibration Error (ACE) with 8 bins
            ace_8 = uncertainty_metrics.ace(true_lables_um, predictions_um, num_bins=8)
            res_list.append(float(ace_8))

            # Compute Expected Calibration Error (ACE) with 16 bins
            ace_16 = uncertainty_metrics.ace(true_lables_um, predictions_um, num_bins=16)
            res_list.append(float(ace_16))

            # Create a new row in the results dataframe from the results list
            row_results_df = pd.DataFrame([res_list], columns=results_df.columns)

            # Append the new results row to the main results dataframe
            results_df = pd.concat([results_df, row_results_df], ignore_index=True)

            # Clear res_list and initialize it with cancer type and data type for the next iteration
            res_list.clear()
            res_list.append(cancer_type)
            res_list.append(data_type)

            # Increment the outer loop index for tracking cross-validation iterations
            outer_loop_index += 1
            inner_loop_index = 1

            # Compute the mean and standard deviation of test balanced accuracy
            test_bal_acc_mean = (sum(test_bal_acc_list) / len(test_bal_acc_list))
            test_bal_acc_std = np.std(test_bal_acc_list)

            # Clear res_list and append the current cancer type
            res_list.clear()
            res_list.append(cancer_type)

            # Clear res_list before moving to the next cancer type
            res_list.clear()

            # Save the final results dataframe to a TSV file
            results_df.to_csv(f'...', sep='\t')
