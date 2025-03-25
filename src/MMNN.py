# Import necessary libraries
import torch
torch.manual_seed(0)  # Set a manual seed for reproducibility
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
np.random.seed(0)  # Set NumPy seed for reproducibility
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from mrmr import mrmr_classif
import uncertainty_metrics.numpy as uncertainty_metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import balanced_accuracy_score

# Set the device to GPU if available, otherwise use CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define the Multimodal Neural Network (MMNN) model
class MMNN(nn.Module):
    def __init__(self, R_input_size, m_input_size, R_layer_size, m_layer_size, 
                 fusion_layer_size, R_dropout_p, m_dropout_p, fusion_dropout_p):
        super().__init__()

        # Define the RNA-seq (R) branch of the network
        self.R_NN_layout = nn.Sequential(
            nn.Linear(R_input_size, R_layer_size),
            nn.BatchNorm1d(R_layer_size),
            nn.Tanh(),

            nn.Linear(R_layer_size, R_layer_size // 2),
            nn.BatchNorm1d(R_layer_size // 2),
            nn.Tanh(),
            nn.Dropout(R_dropout_p),

            nn.Linear(R_layer_size // 2, R_layer_size // 4),
            nn.BatchNorm1d(R_layer_size // 4),
            nn.Tanh(),
            nn.Dropout(R_dropout_p),

            nn.Linear(R_layer_size // 4, R_layer_size // 4)
            # No activation function here since BCEWithLogitsLoss will be used
        )

        # Define the miRNA-seq (m) branch of the network
        self.m_NN_layout = nn.Sequential(
            nn.Linear(m_input_size, m_layer_size),
            nn.BatchNorm1d(m_layer_size),
            nn.Tanh(),

            nn.Linear(m_layer_size, m_layer_size // 2),
            nn.BatchNorm1d(m_layer_size // 2),
            nn.Tanh(),
            nn.Dropout(m_dropout_p),

            nn.Linear(m_layer_size // 2, m_layer_size // 4),
            nn.BatchNorm1d(m_layer_size // 4),
            nn.Tanh(),
            nn.Dropout(m_dropout_p),

            nn.Linear(m_layer_size // 4, m_layer_size // 4)
            # No activation function since BCEWithLogitsLoss will be used
        )

        # Calculate the input size for the fusion layer
        fusion_input_size = R_layer_size // 4 + m_layer_size // 4

        # Define the fusion layer to combine features from both modalities
        self.fusion_layout = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_layer_size),
            nn.BatchNorm1d(fusion_layer_size),
            nn.Tanh(),

            nn.Linear(fusion_layer_size, fusion_layer_size // 2),
            nn.BatchNorm1d(fusion_layer_size // 2),
            nn.Tanh(),
            nn.Dropout(fusion_dropout_p),

            nn.Linear(fusion_layer_size // 2, fusion_layer_size // 4),
            nn.BatchNorm1d(fusion_layer_size // 4),
            nn.Tanh(),
            nn.Dropout(fusion_dropout_p),

            nn.Linear(fusion_layer_size // 4, fusion_layer_size // 4),
            nn.BatchNorm1d(fusion_layer_size // 4),
            nn.Tanh(),
            nn.Dropout(fusion_dropout_p),

            nn.Linear(fusion_layer_size // 4, 1)
            # No activation function since BCEWithLogitsLoss will be used
        )

    # Define forward pass for the model
    def forward(self, xR, xm):
        logits_R = self.R_NN_layout(xR)  # Process RNA-seq data
        logits_m = self.m_NN_layout(xm)  # Process miRNA-seq data

        # Concatenate features from both branches
        comb_input = torch.cat([logits_R, logits_m], dim=1)

        # Pass concatenated features through the fusion network
        logits = self.fusion_layout(comb_input)

        return logits

    # Function to adjust dropout probability dynamically
    def set_dropout_p(self, dropout_p):
        for module in self.R_NN_layout:
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

        for module in self.m_NN_layout:
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

        for module in self.fusion_layout:
            if isinstance(module, nn.Dropout):
                module.p = dropout_p


# Function to enable dropout during inference for Monte Carlo Dropout
def enable_dropout(model: MMNN):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# Function for preprocessing and feature selection
def preprocessing_PL(R_X_train, m_X_train, R_X_test, m_X_test, y_train, y_test, R_input_size, m_input_size):
    
    # Feature selection using Chi-Square test
    R_chi2_selector = SelectKBest(score_func=chi2, k=R_input_size * 2)
    R_X_train_ch2 = R_chi2_selector.fit_transform(R_X_train, y_train)
    R_X_test_ch2 = R_chi2_selector.transform(R_X_test)

    m_chi2_selector = SelectKBest(score_func=chi2, k=m_input_size * 2)
    m_X_train_ch2 = m_chi2_selector.fit_transform(m_X_train, y_train)
    m_X_test_ch2 = m_chi2_selector.transform(m_X_test)

    # Feature selection using Minimum Redundancy Maximum Relevance (mRMR)
    R_selected_features = mrmr_classif(X=pd.DataFrame(R_X_train_ch2), y=pd.DataFrame(y_train), K=R_input_size, show_progress=False)
    R_X_train_sel = R_X_train_ch2[:, R_selected_features]
    R_X_test_sel = R_X_test_ch2[:, R_selected_features]

    m_selected_features = mrmr_classif(X=pd.DataFrame(m_X_train_ch2), y=pd.DataFrame(y_train), K=m_input_size, show_progress=False)
    m_X_train_sel = m_X_train_ch2[:, m_selected_features]
    m_X_test_sel = m_X_test_ch2[:, m_selected_features]

    # Standardize features
    R_scaler = StandardScaler()
    m_scaler = StandardScaler()

    R_X_train_sel = R_scaler.fit_transform(R_X_train_sel)
    R_X_test_sel = R_scaler.transform(R_X_test_sel)

    m_X_train_sel = m_scaler.fit_transform(m_X_train_sel)
    m_X_test_sel = m_scaler.transform(m_X_test_sel)

    # Convert NumPy arrays to PyTorch tensors
    R_X_train_T = torch.tensor(R_X_train_sel, dtype=torch.float32)
    R_X_test_T = torch.tensor(R_X_test_sel, dtype=torch.float32)

    m_X_train_T = torch.tensor(m_X_train_sel, dtype=torch.float32)
    m_X_test_T = torch.tensor(m_X_test_sel, dtype=torch.float32)

    y_train_T = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_T = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return R_X_train_T.to(device), R_X_test_T.to(device), m_X_train_T.to(device), m_X_test_T.to(device), y_train_T.to(device), y_test_T.to(device)

# Trains the multimodal neural network (MMNN) model using the provided data
def train_model(model: MMNN, optimizer, loss_fn, 
                R_X_train_T, R_X_test_T, 
                m_X_train_T, m_X_test_T, 
                y_train_T, y_test_T, 
                hyperparameters, plot_info):

    train_losses = []  # Stores training loss per epoch
    val_accuracies = []  # Stores validation accuracy per epoch

    for epoch in range(hyperparameters['epochs']):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track cumulative loss for this epoch

        # Iterate through training data in batches
        for batch in range(0, len(R_X_train_T), hyperparameters['batch_size']):
            model.train()
            # Extract batch of RNA-seq, miRNA-seq, and labels
            R_X_train_T_batch = R_X_train_T[batch: batch + hyperparameters['batch_size']]
            m_X_train_T_batch = m_X_train_T[batch: batch + hyperparameters['batch_size']]
            y_train_T_batch = y_train_T[batch: batch + hyperparameters['batch_size']]

            # Proceed only if batch size is valid
            if R_X_train_T_batch.size(0) > 1:
                optimizer.zero_grad()  # Reset gradients
                outputs = model(R_X_train_T_batch, m_X_train_T_batch)  # Forward pass
                loss = loss_fn(outputs, y_train_T_batch)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                epoch_loss += loss.item()  # Accumulate loss

        # If plot_info is provided, compute and store loss/accuracy for visualization
        if plot_info is not None:
            avg_epoch_loss = epoch_loss / (len(R_X_train_T) / hyperparameters['batch_size'])
            train_losses.append(avg_epoch_loss)
            temp_val_acc = eval_model(model, R_X_test_T, m_X_test_T, y_test_T)
            val_accuracies.append(temp_val_acc)

    # Plot loss and accuracy if plot_info is provided
    if plot_info is not None:
        plot_loss_balAcc(train_losses, val_accuracies, hyperparameters, plot_info)

    return model

# Evaluates the trained model on the test dataset
def eval_model(model: MMNN, R_X_test_T: torch.Tensor, m_X_test_T: torch.Tensor, y_test_T: torch.Tensor):
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        y_pred_val = model(R_X_test_T, m_X_test_T)  # Get predictions
        y_pred_val = torch.sigmoid(y_pred_val)  # Convert logits to probabilities
        y_pred_val = (y_pred_val > 0.5).float()  # Convert to binary predictions
        test_bal_acc = balanced_accuracy_score(y_test_T.cpu().numpy(), y_pred_val.cpu().numpy())  # Compute balanced accuracy

    return test_bal_acc

# Plots and saves training loss and balanced accuracy over epochs
def plot_loss_balAcc(train_losses, val_accuracies, hyperparameters, plot_info):

    epochs_range = range(1, hyperparameters['epochs'] + 1)
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Plot balanced accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, marker='o', color='orange', label="Balanced accuracy")
    plt.xlabel("Epochs")
    plt.ylabel('Balanced accuracy')
    plt.title("Balanced accuracy Over Epochs")
    plt.legend()

    # Save plot with provided metadata
    save_path = f'...'
    plt.savefig(f'{save_path}/{plot_info[0]}_MM_{plot_info[1]}_{plot_info[2]}_{plot_info[3]}.png')
    plt.close('all')

# Generates a dictionary of randomly sampled hyperparameters
def hyperparameters(np_seed):

    np.random.seed(np_seed)  # Set seed for reproducibility

    hyperparameters_dict = dict(
        lr=10 ** np.random.choice(np.arange(-6, -2, 0.5)),  # Learning rate
        wd=10 ** np.random.choice(np.arange(-5, -1, 0.5)),  # Weight decay
        epochs=np.random.choice(np.arange(5, 201, 5)),  # Number of epochs
        batch_size=2 ** np.random.randint(3, 6),  # Batch size

        # Dropout probabilities for different parts of the model
        R_dropout_p=np.random.choice(np.arange(0.1, 0.8, 0.05)),
        m_dropout_p=np.random.choice(np.arange(0.1, 0.8, 0.05)),
        fusion_dropout_p=np.random.choice(np.arange(0.1, 0.8, 0.05)),

        # Layer sizes for different model components
        R_input_size=np.random.choice(np.arange(50, 5001, 50)),
        R_layer_size=np.random.choice(np.arange(50, 5001, 50)),
        m_input_size=np.random.choice(np.arange(10, 532, 5)),
        m_layer_size=np.random.choice(np.arange(20, 501, 10)),
        fusion_layer_size=np.random.choice(np.arange(50, 5001, 50))
    )

    return hyperparameters_dict

# Performs Monte Carlo Dropout to estimate uncertainty in predictions
def monte_carlo_dropout(model: MMNN, R_X_test_T, m_X_test_T, forward_passes):

    model.eval()
    enable_dropout(model)  # Enable dropout during inference
    predictions = []

    for _ in range(forward_passes):
        with torch.no_grad():
            output = torch.sigmoid(model(R_X_test_T, m_X_test_T))
            predictions.append(output.unsqueeze(0))

    predictions_T = torch.cat(predictions, dim=0)
    mean_predictions_T = predictions_T.mean(dim=0)
    variance_T = predictions_T.var(dim=0)
    mean_variance = variance_T.mean()

    return mean_predictions_T, mean_variance, variance_T, predictions_T

def calculate_uncertainty_ratio(mean_predictions_T, variance_T, y_test_T):
    # Convert the predicted probabilities to binary predictions (1 if > 0.5, else 0)
    binary_predictions = (mean_predictions_T > 0.5).float()

    # Initialize tensors to hold masks for correct and incorrect predictions
    correct_mask = torch.zeros_like(binary_predictions)
    incorrect_mask = torch.zeros_like(binary_predictions)

    # Loop through each prediction and compare with the true labels
    for i in range(len(binary_predictions)):
        if binary_predictions[i] == y_test_T[i]:  # If prediction is correct
            correct_mask[i] = 1.0  # Mark as correct
            incorrect_mask[i] = 0.0  # Mark as incorrect
        else:  # If prediction is incorrect
            correct_mask[i] = 0.0  # Mark as incorrect
            incorrect_mask[i] = 1.0  # Mark as correct

    # Calculate the mean variance for correct predictions
    correct_variance = (variance_T * correct_mask).sum() / correct_mask.sum()

    # Calculate the mean variance for incorrect predictions
    incorrect_variance = (variance_T * incorrect_mask).sum() / incorrect_mask.sum()
    
    # Calculate the uncertainty ratio as the ratio of incorrect to correct variance
    uncertainty_ratio = incorrect_variance / correct_variance
    
    return uncertainty_ratio


# Computes entropy of predictions for uncertainty estimation
def calculate_entropy(mean_predictions_T):

    mean_predictions_T = torch.clamp(mean_predictions_T, min=1e-8, max=1 - 1e-8)
    entropy = -mean_predictions_T * torch.log(mean_predictions_T) - (1 - mean_predictions_T) * torch.log(1 - mean_predictions_T)

    return entropy.to(torch.float32)


# List of cancer types to analyze
cancer_type_list = ['OV', 'BRCA']
res_list = []  # Stores results for each cancer type

# Define cross-validation and iteration parameters
n_iter = 60  # Number of iterations for hyperparameter tuning
outer_splits = 4  # Number of splits for outer cross-validation
inner_splits = 3  # Number of splits for inner cross-validation
np_seed = 0  # Seed for reproducibility

# Path to the dataset
dataset_path = '...'

# DataFrame to store results
results_df = pd.DataFrame(columns=['dataset', 'Ball_acc', '0.2_mean_variance', '0.2_uncertainty_ratio', '0.2_mean_entropy', 
                                                            '0.5_mean_variance', '0.5_uncertainty_ratio', '0.5_mean_entropy', 
                                                            '0.8_mean_variance', '0.8_uncertainty_ratio', '0.8_mean_entropy', '8_ACE', '16_ACE'])

# Loop through each cancer type in the list
for cancer_type in cancer_type_list:

    res_list.append(cancer_type)  # Append cancer type to results list

    # Define file paths for RNA and miRNA data
    R_path = f'{dataset_path}/{cancer_type}_RNA_taining_data_no_low_counts.tsv'
    m_path = f'{dataset_path}/{cancer_type}_miRNA_taining_data_no_low_counts.tsv'

    # Load the RNA and miRNA data
    R_data = pd.read_csv(R_path, sep='\t')
    m_data = pd.read_csv(m_path, sep='\t')

    # Extract feature matrices and class labels
    R_X = R_data.drop(columns='class').values  # RNA features
    m_X = m_data.drop(columns='class').values  # miRNA features
    y = R_data['class'].values  # Class labels

    test_bal_acc_list = []  # Stores balanced accuracy results

    outer_loop_index = 1  # Tracks the outer cross-validation loop

    # Outer cross-validation loop
    outer_cv = StratifiedKFold(n_splits=outer_splits)
    for train_val_idx, test_idx in outer_cv.split(R_X, y):

        start = datetime.now()

        print(f'\n#################\nCancer type: {cancer_type}\nLoop n {outer_loop_index}:{outer_splits}\nStarting at {start}')
        
        # Split data into training/validation and test sets
        R_X_train_val  = R_X[train_val_idx]
        R_X_test = R_X[test_idx]

        m_X_train_val  = m_X[train_val_idx]
        m_X_test = m_X[test_idx]
        
        y_train_val = y[train_val_idx]
        y_test = y[test_idx]

        mean_best_val_bal_acc = 0  # Track best validation accuracy
        best_val_par = None  # Store best hyperparameters

        # Hyperparameter tuning loop
        for k in range(n_iter):
            hyperparameters_list = hyperparameters(np_seed)  # Generate random hyperparameters
            np_seed += 1  # Increment seed for reproducibility
            temp_val_bal_acc = []  # Store validation accuracy for different hyperparameters
            inner_loop_index = 1  # Track inner cross-validation loop

            # Inner cross-validation loop for model selection
            inner_cv = StratifiedKFold(n_splits=inner_splits)
            for train_idx, val_idx in inner_cv.split(R_X_train_val, y_train_val):

                # Split data into training and validation sets
                R_X_train  = R_X_train_val[train_idx]
                R_X_val = R_X_train_val[val_idx]

                m_X_train  = m_X_train_val[train_idx]
                m_X_val = m_X_train_val[val_idx]

                y_train  = y_train_val[train_idx]
                y_val = y_train_val[val_idx]

                # Preprocess the data
                (R_X_train_T, 
                  R_X_val_T, 
                  m_X_train_T,
                  m_X_val_T, 
                  y_train_T,
                  y_val_T) = preprocessing_PL(R_X_train, m_X_train, R_X_val, m_X_val, y_train, y_val, 
                                              hyperparameters_list['R_input_size'], hyperparameters_list['m_input_size'])

                # Initialize the model with current hyperparameters
                inner_model = MMNN(hyperparameters_list['R_input_size'], hyperparameters_list['m_input_size'],
                                   hyperparameters_list['R_layer_size'], hyperparameters_list['m_layer_size'], 
                                   hyperparameters_list['fusion_layer_size'], hyperparameters_list['R_dropout_p'], 
                                   hyperparameters_list['m_dropout_p'], hyperparameters_list['fusion_dropout_p']).to(device)
        
                # Define loss function and optimizer
                inner_loss_fn = nn.BCEWithLogitsLoss()
                inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=hyperparameters_list['lr'], weight_decay=hyperparameters_list['wd'])
                
                # Train the model
                inner_trained_model = train_model(inner_model,
                                                  inner_optimizer,
                                                  inner_loss_fn,
                                                  R_X_train_T,
                                                  R_X_val_T,
                                                  m_X_train_T,
                                                  m_X_val_T,
                                                  y_train_T,
                                                  y_val_T,
                                                  hyperparameters_list,
                                                  plot_info=None)
                
                # Evaluate the model on validation data
                bal_acc = eval_model(inner_trained_model, R_X_val_T, m_X_val_T, y_val_T)
                temp_val_bal_acc.append(bal_acc)  # Store validation accuracy
                inner_loop_index += 1

            # Compute mean validation accuracy across inner splits
            mean_temp_val_bal_acc = sum(temp_val_bal_acc) / len(temp_val_bal_acc)
            temp_val_bal_acc_sd = np.std(temp_val_bal_acc)  # Standard deviation of accuracy

            # Update best hyperparameters if the current set performs better
            if mean_temp_val_bal_acc > mean_best_val_bal_acc:
                mean_best_val_bal_acc = mean_temp_val_bal_acc
                best_val_par = hyperparameters_list

        # Final testing with the best selected hyperparameters
        (R_X_train_T, 
         R_X_test_T,
         m_X_train_T, 
         m_X_test_T,
         y_train_T,
         y_test_T) = preprocessing_PL(R_X_train_val,
                                      m_X_train_val,
                                      R_X_test,
                                      m_X_test,
                                      y_train_val,
                                      y_test,
                                      best_val_par['R_input_size'],
                                      best_val_par['m_input_size'])
        
        # Initialize the final model with best hyperparameters
        outer_model = MMNN(best_val_par['R_input_size'], best_val_par['m_input_size'],
                           best_val_par['R_layer_size'], best_val_par['m_layer_size'], 
                           best_val_par['fusion_layer_size'], best_val_par['R_dropout_p'], 
                           best_val_par['m_dropout_p'], best_val_par['fusion_dropout_p']).to(device)
        
        # Define optimizer and loss function
        outer_optimizer = optim.Adam(outer_model.parameters(), lr=best_val_par['lr'], weight_decay=best_val_par['wd'])
        outer_loss_fn = nn.BCEWithLogitsLoss()

        # Train the final model on the full training set
        trained_outer_model = train_model(outer_model,
                                          outer_optimizer,
                                          outer_loss_fn,
                                          R_X_train_T,
                                          R_X_test_T,
                                          m_X_train_T,
                                          m_X_test_T,
                                          y_train_T,
                                          y_test_T,
                                          best_val_par,
                                          plot_info=['TEST', cancer_type, outer_loop_index, '', ''])
        
        # Evaluate the trained model on the test set
        test_bal_acc = eval_model(trained_outer_model, R_X_test_T, m_X_test_T, y_test_T)
        res_list.append(float(test_bal_acc))  # Store test accuracy
        # MCDo (Monte Carlo Dropout)
        drop_p = [0.2, 0.5, 0.8]  # List of dropout probabilities to test
        foreward_passes = 50  # Number of forward passes for Monte Carlo Dropout

        # Loop through each dropout probability value
        for p in drop_p:

            # Set the dropout probability for the model
            trained_outer_model.set_dropout_p(p)

            # Perform Monte Carlo Dropout: compute mean predictions, variance, and uncertainty metrics
            mean_predictions_T, mean_variance, variance_T, predictions_T = monte_carlo_dropout(trained_outer_model, R_X_test_T, m_X_test_T, foreward_passes)

            # Calculate uncertainty ratio and entropy
            uncertainty_ratio = calculate_uncertainty_ratio(mean_predictions_T, variance_T, y_test_T)
            entropy = calculate_entropy(mean_predictions_T)
            mean_entropy = entropy.mean()  # Average entropy across predictions

            # Append the calculated metrics to the result list
            res_list.append(float(mean_variance))
            res_list.append(float(uncertainty_ratio))
            res_list.append(float(mean_entropy))   

            # Prepare predictions for plotting
            mean_predictions_for_plot = mean_predictions_T.cpu().detach().numpy().flatten()

            # Plot histogram of predicted confidence values
            plt.hist(mean_predictions_for_plot, bins=10, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.title('Histogram of Predicted Confidences')

            # Save the confidence plot as an image
            plt.savefig(f'...')
            plt.close('all')  # Close the plot to free up memory

        # Switch the model to evaluation mode
        trained_outer_model.eval()

        # Disable gradient calculation during evaluation
        with torch.no_grad():
            # Perform predictions using the trained model
            um_predictions_T = torch.sigmoid(trained_outer_model(R_X_test_T, m_X_test_T))

        # Convert true labels to a flat numpy array
        true_lables_um = y_test_T.cpu().detach().numpy().flatten().astype(int)

        # Prepare the prediction probabilities for both classes (class 0 and class 1)
        predictions_um = np.hstack([1 - um_predictions_T.cpu().detach().numpy(),  # Probability of class 0
                                    um_predictions_T.cpu().detach().numpy()])    # Probability of class 1

        # Calculate the ACE (Adaptive Calibration Error) for 8 bins
        ace_8 = uncertainty_metrics.ace(true_lables_um, predictions_um, num_bins=8)
        res_list.append(float(ace_8))

        # Calculate the ACE for 16 bins
        ace_16 = uncertainty_metrics.ace(true_lables_um, predictions_um, num_bins=16)
        res_list.append(float(ace_16))

        # Store the results for this iteration in a DataFrame
        row_results_df = pd.DataFrame([res_list], columns=results_df.columns)

        # Append the row to the main results DataFrame
        results_df = pd.concat([results_df, row_results_df], ignore_index=True)

        # Clear the results list and add the cancer type for the next iteration
        res_list.clear()
        res_list.append(cancer_type)

        # Log the end time of the current iteration
        end = datetime.now()
        print(f'Finishing at. {end}\nRemaining time: {end-start}\nTest balanced accuracy: {test_bal_acc}\n#################\n')

        # Increment the outer and inner loop indices for tracking progress
        outer_loop_index += 1
        inner_loop_index = 1

    # Clear the results list after finishing the current cancer type
    res_list.clear()

# Save the results DataFrame to a TSV file
results_df.to_csv(f'....', sep='\t')
