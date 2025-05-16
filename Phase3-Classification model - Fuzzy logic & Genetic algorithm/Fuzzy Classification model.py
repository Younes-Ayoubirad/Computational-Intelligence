import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load the dataset
df = pd.read_csv('data.csv', sep=';')

# Impute missing values: mean for numerical, mode for categorical
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode categorical features, dropping the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df.drop('Target', axis=1), drop_first=True)

# Map target variable to numerical values
target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
df_encoded['Target'] = df['Target'].map(target_mapping)

# Visualize the distribution of target classes
sns.countplot(x='Target', data=df_encoded)
plt.title('Distribution of Student Target Classes')
plt.xticks(ticks=[0, 1, 2], labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xlabel('Student Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Visualize the correlation heatmap of the encoded features
plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Prepare features (X) and target (y)
X = df_encoded.drop('Target', axis=1)
y = df_encoded['Target']

# Split data into training and testing sets (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define triangular membership function for fuzzification
def triangular_membership(x, a, b, c):
    """
    Computes the triangular membership function.
    x: input value or array
    a, b, c: parameters defining the triangle (a <= b <= c)
    """
    # Handle cases where a, b, or c might be equal to avoid division by zero
    if a == c: # Covers a=b=c case, results in a crisp set at point b
        return np.ones_like(x) if np.all(x == a) else np.zeros_like(x)

    b_minus_a = b - a if b - a != 0 else 1e-8 # Small epsilon to avoid division by zero
    c_minus_b = c - b if c - b != 0 else 1e-8 # Small epsilon to avoid division by zero

    # Calculate left and right slopes of the triangle
    left_slope = (x - a) / b_minus_a
    right_slope = (c - x) / c_minus_b

    # Combine slopes and ensure membership is between 0 and 1
    return np.maximum(np.minimum(left_slope, right_slope), 0)

# Fuzzify continuous features using triangular membership functions (Low, Medium, High)
def fuzzify_continuous_feature(feature_values, feature_name):
    """
    Fuzzifies a continuous feature into Low, Medium, and High fuzzy sets.
    feature_values: pandas Series of the continuous feature
    feature_name: string name of the feature
    """
    # Define parameters for Low, Medium, High membership functions
    low = triangular_membership(feature_values, feature_values.min(), feature_values.min(), feature_values.mean())
    medium = triangular_membership(feature_values, feature_values.min(), feature_values.mean(), feature_values.max())
    high = triangular_membership(feature_values, feature_values.mean(), feature_values.max(), feature_values.max())

    # Create a DataFrame for the fuzzy sets
    fuzzy_df = pd.DataFrame({
        f'{feature_name}_Low': low,
        f'{feature_name}_Medium': medium,
        f'{feature_name}_High': high
    })
    return fuzzy_df

# Fuzzify binary features (No, Yes)
def fuzzify_binary_feature(feature_series, feature_name):
    """
    Fuzzifies a binary feature into No and Yes fuzzy sets.
    feature_series: pandas Series of the binary feature (0 or 1)
    feature_name: string name of the feature
    """
    fuzzy_df = pd.DataFrame({
        f'{feature_name}_No': 1 - feature_series, # Membership for 'No'
        f'{feature_name}_Yes': feature_series     # Membership for 'Yes'
    })
    return fuzzy_df

# Define continuous and binary features for fuzzification
cont_features = ['Admission grade', 'Unemployment rate','Application mode','Course','Previous qualification'
    ,'Curricular units 1st sem (enrolled)','Curricular units 2nd sem (grade)','Curricular units 2nd sem (enrolled)']
binary_features = ['Gender', 'Debtor', 'Scholarship holder'] # Assuming these were one-hot encoded from original binary

# Fuzzify continuous features for the training set
fuzzy_cont_train = [fuzzify_continuous_feature(X_train[feat], feat) for feat in cont_features]
fuzzy_cont_train_df = pd.concat(fuzzy_cont_train, axis=1)

# Fuzzify binary features for the training set
fuzzy_bin_train = [fuzzify_binary_feature(X_train[feat], feat) for feat in binary_features]
fuzzy_bin_train_df = pd.concat(fuzzy_bin_train, axis=1)

# Combine fuzzified continuous and binary features for the training set
X_train_fuzzy = pd.concat([fuzzy_cont_train_df, fuzzy_bin_train_df], axis=1)

# Fuzzify continuous features for the test set
fuzzy_cont_test = [fuzzify_continuous_feature(X_test[feat], feat) for feat in cont_features]
fuzzy_cont_test_df = pd.concat(fuzzy_cont_test, axis=1)

# Fuzzify binary features for the test set
fuzzy_bin_test = [fuzzify_binary_feature(X_test[feat], feat) for feat in binary_features]
fuzzy_bin_test_df = pd.concat(fuzzy_bin_test, axis=1)

# Combine fuzzified continuous and binary features for the test set
X_test_fuzzy = pd.concat([fuzzy_cont_test_df, fuzzy_bin_test_df], axis=1)


# Plot triangular membership functions for continuous features
for feat in cont_features:
    x_vals = np.linspace(df[feat].min(), df[feat].max(), 1000) # Generate values for plotting
    # Calculate membership degrees for Low, Medium, High
    low_membership = triangular_membership(x_vals, x_vals.min(), x_vals.min(), x_vals.mean())
    medium_membership = triangular_membership(x_vals, x_vals.min(), x_vals.mean(), x_vals.max())
    high_membership = triangular_membership(x_vals, x_vals.mean(), x_vals.max(), x_vals.max())

    plt.figure()
    plt.plot(x_vals, low_membership, label='Low')
    plt.plot(x_vals, medium_membership, label='Medium')
    plt.plot(x_vals, high_membership, label='High')
    plt.title(f'Triangular Membership Functions for {feat}')
    plt.xlabel(feat)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Extract fuzzy rules using the Wang-Mendel method
def extract_rules(X_fuzzy, y_target):
    """
    Extracts fuzzy rules from fuzzified data.
    X_fuzzy: DataFrame of fuzzified input features
    y_target: Series of target labels
    """
    rules = {} # Dictionary to store rules and their counts/labels
    # Iterate through each sample to form rules
    for i in range(len(X_fuzzy)):
        condition = tuple(X_fuzzy.iloc[i]) # Antecedent of the rule
        label = y_target.iloc[i]          # Consequent of the rule

        # If condition already exists, update its count and labels
        if condition in rules:
            rules[condition]['count'] += 1
            rules[condition]['labels'].append(label)
        # Otherwise, add new rule
        else:
            rules[condition] = {'count': 1, 'labels': [label]}

    final_rules = []
    # Determine the final label and confidence for each rule
    for condition, data in rules.items():
        label_counts = pd.Series(data['labels']).value_counts() # Count occurrences of each label for the condition
        final_label = label_counts.idxmax()                    # Most frequent label
        confidence = label_counts.max() / sum(label_counts)    # Confidence of the rule
        final_rules.append((condition, final_label, confidence))
    return final_rules

# Extract rules from the fuzzified training data
rules = extract_rules(X_train_fuzzy, y_train)

# Print some sample extracted rules
print("\nSample Extracted Rules:")
for i, (condition, label, conf) in enumerate(rules[:5]): # Show first 5 rules
    print(f"Rule {i + 1}:")
    print("IF:")
    for j, val in enumerate(condition):
        print(f"  {X_train_fuzzy.columns[j]} = {val}")
    print(f"THEN: Class = {['Dropout', 'Enrolled', 'Graduate'][label]}") # Map label back to original name
    print(f"Confidence Weight: {conf:.2f}\n")
print(f"Total number of extracted rules: {len(rules)}")


import random
# Note: numpy as np is already imported
from concurrent.futures import ThreadPoolExecutor # For parallel processing

# Genetic Algorithm parameters
POP_SIZE = 20          # Number of individuals (rule sets) in the population
NUM_GENERATIONS = 50   # Number of generations to run the GA
MUTATION_RATE = 0.2    # Probability of mutating a rule in a chromosome
CROSSOVER_RATE = 0.8   # Probability of performing crossover
ELITE_COUNT = 2        # Number of best individuals to carry over to the next generation
MAX_WORKERS = 100      # Maximum number of threads for parallel evaluation

# Evaluate a given set of rules on a validation set
def evaluate_rule_set(rule_set, X_val, y_val):
    """
    Evaluates the fitness of a rule set (chromosome).
    rule_set: A list of rules (tuples of condition, label, confidence)
    X_val: Validation features (fuzzified)
    y_val: Validation target labels
    """
    correct_predictions = 0
    # Iterate through each sample in the validation set
    for i in range(len(X_val)):
        sample = tuple(X_val.iloc[i]) # Current sample's fuzzified features
        true_label = y_val.iloc[i]    # True label of the current sample
        scores = {0: 0, 1: 0, 2: 0}   # Initialize scores for each class (Dropout, Enrolled, Graduate)

        # For each rule in the rule set, calculate its contribution
        for condition, label, conf in rule_set:
            degree_of_fulfillment = 1.0
            # Calculate the similarity (degree of fulfillment) of the sample to the rule's condition
            for j, val in enumerate(sample):
                similarity = 1 - abs(condition[j] - val) # Simple similarity measure
                degree_of_fulfillment *= max(0, similarity) # Aggregate similarity (min operator for fuzzy AND)
            scores[label] += degree_of_fulfillment * conf # Add weighted score to the corresponding class

        # Predict the label based on the highest score
        if max(scores.values()) > 0:
            predicted_label = max(scores, key=scores.get)
        else:
            # If no rule fires strongly, make a random choice (or handle as per strategy)
            predicted_label = np.random.choice([0, 1, 2])

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_val)
    # Penalty for having too many rules to encourage simpler models
    penalty = 0.0001 * len(rule_set)
    return accuracy - penalty # Fitness score

# Evaluate the entire population in parallel
def evaluate_population_parallel(population, X_val, y_val, max_workers=4):
    """
    Evaluates all individuals in the population using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each rule set evaluation as a separate task
        futures = [executor.submit(evaluate_rule_set, chromosome, X_val, y_val) for chromosome in population]
        # Collect results as they complete
        return [f.result() for f in futures]

# Select parents using tournament selection
def tournament_selection(population, scores, k=3):
    """
    Selects a parent from the population using k-tournament selection.
    k: Number of individuals participating in the tournament
    """
    # Randomly select k individuals for the tournament
    selected_indices = random.sample(range(len(population)), k)
    tournament_participants = [(population[i], scores[i]) for i in selected_indices]
    # Sort participants by score (descending)
    tournament_participants.sort(key=lambda x: x[1], reverse=True)
    # The winner is the one with the highest score
    return tournament_participants[0][0]

# Perform crossover between two parents to create offspring
def crossover(parent1, parent2):
    """
    Performs single-point crossover between two parent chromosomes.
    """
    # If random number is greater than crossover rate, no crossover occurs, return parents as children
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:] # Return copies

    # Determine the crossover point (ensure it's within bounds)
    min_len = min(len(parent1), len(parent2))
    if min_len <= 1: # Not enough length for crossover
        return parent1[:], parent2[:]

    cut_point = random.randint(1, min_len - 1)
    # Create children by swapping segments
    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]
    return child1, child2

# Mutate a chromosome by randomly changing one of its rules
def mutate(chromosome, all_rules_pool):
    """
    Mutates a chromosome by replacing one rule with a random rule from the initial pool.
    all_rules_pool: The complete set of initially extracted rules
    """
    if random.random() < MUTATION_RATE and len(chromosome) > 0:
        mutation_index = random.randint(0, len(chromosome) - 1) # Index of the rule to mutate
        new_rule = random.choice(all_rules_pool)             # Select a new random rule
        chromosome[mutation_index] = new_rule
    return chromosome

# Main Genetic Algorithm function for optimizing the rule set
def genetic_algorithm_optimized(initial_rules_pool, X_val, y_val):
    """
    Optimizes the set of fuzzy rules using a genetic algorithm.
    initial_rules_pool: The full set of rules extracted by Wang-Mendel
    X_val: Validation features (fuzzified)
    y_val: Validation target labels
    """
    # Initialize population: each individual is a random sample of rules
    # Ensure rule_length for individuals is reasonable (e.g., max 15 or length of initial_rules_pool if smaller)
    rule_length_for_individuals = min(15, len(initial_rules_pool))
    if rule_length_for_individuals == 0 and len(initial_rules_pool) > 0: # If initial pool is small but not empty
        rule_length_for_individuals = len(initial_rules_pool)
    elif rule_length_for_individuals == 0: # If initial pool is empty
        print("Warning: Initial rule pool is empty. GA cannot proceed effectively.")
        return []


    population = [random.sample(initial_rules_pool, rule_length_for_individuals) for _ in range(POP_SIZE)]


    # Evolution loop
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness of each individual in the population
        scores = evaluate_population_parallel(population, X_val, y_val, MAX_WORKERS)
        print(f"Generation {generation + 1}: Best Fitness = {max(scores):.3f}")

        # Sort population by fitness (descending)
        sorted_population_with_scores = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        sorted_population = [chromosome for score, chromosome in sorted_population_with_scores]

        # Elitism: carry over the best individuals to the next generation
        next_generation = sorted_population[:ELITE_COUNT]

        # Generate the rest of the new population through selection, crossover, and mutation
        while len(next_generation) < POP_SIZE:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([
                mutate(child1, initial_rules_pool),
                mutate(child2, initial_rules_pool)
            ])
        population = next_generation[:POP_SIZE] # Ensure population size is maintained

    # After all generations, evaluate the final population one last time
    final_scores = evaluate_population_parallel(population, X_val, y_val, MAX_WORKERS)
    best_individual_index = np.argmax(final_scores)
    best_rule_set = population[best_individual_index]
    print(f"\nBest final fitness: {final_scores[best_individual_index]:.3f}")
    return best_rule_set

# Run the genetic algorithm to get the best set of rules
best_rules = genetic_algorithm_optimized(rules, X_val=X_test_fuzzy, y_val=y_test)

# Print some of the final optimized rules
print("\nSome Final Optimized Rules:")
for i, (condition, label, conf) in enumerate(best_rules[:5]): # Show first 5 rules
    print(f"Rule {i+1}:")
    for j, val in enumerate(condition):
        print(f"  {X_train_fuzzy.columns[j]} = {val}") # Use X_train_fuzzy for column names as rules were derived from it
    print(f"→ Class: {['Dropout', 'Enrolled', 'Graduate'][label]}")
    print(f"→ Confidence Weight: {conf:.2f}\n")

print(f"Number of final selected rules: {len(best_rules)}")


# Perform fuzzy inference using the selected rule set
def fuzzy_inference(selected_rule_set, X_val_fuzzy):
    """
    Performs fuzzy inference to predict class labels for new samples.
    selected_rule_set: The optimized set of fuzzy rules
    X_val_fuzzy: DataFrame of fuzzified input features for samples to predict
    """
    predictions = []
    # Iterate through each sample in the validation/test set
    for i in range(len(X_val_fuzzy)):
        sample = tuple(X_val_fuzzy.iloc[i]) # Current sample's fuzzified features
        scores = {0: 0, 1: 0, 2: 0}         # Initialize scores for each class

        # For each rule in the rule set, calculate its contribution
        for condition, label, conf in selected_rule_set:
            degree_of_fulfillment = 1.0
            # Calculate the similarity of the sample to the rule's condition
            for j, val in enumerate(sample):
                similarity = 1 - abs(condition[j] - val)
                degree_of_fulfillment *= max(0, similarity) # Fuzzy AND (min or product)
            scores[label] += degree_of_fulfillment * conf # Add weighted score

        # Predict the label based on the highest score
        if max(scores.values()) > 0:
            predicted_label = max(scores, key=scores.get)
        else:
            # Handle cases where no rule fires (e.g., random choice or default class)
            predicted_label = np.random.choice([0, 1, 2])
        predictions.append(predicted_label)
    return predictions

# Make predictions on the test set using the best rules
y_pred = fuzzy_inference(best_rules, X_test_fuzzy)

# Calculate and print the overall error rate
errors = sum(1 for i in range(len(y_test)) if y_test.iloc[i] != y_pred[i])
error_rate = (errors / len(y_test)) * 100
print(f"\nOverall Error: {errors} errors out of {len(y_test)} samples ({error_rate:.2f}% error)")


# Evaluate the model's performance
def evaluate_model(y_true, y_pred_labels, title="Model Evaluation"):
    """
    Evaluates the model using various metrics and plots a confusion matrix.
    y_true: True labels
    y_pred_labels: Predicted labels
    title: Title for the evaluation output and plot
    """
    # Distribution of predictions
    pred_counts = pd.Series(y_pred_labels).value_counts().sort_index()
    print(f"\nDistribution of predictions in {title}:")
    for cls_idx, count in pred_counts.items():
        print(f"Class {['Dropout', 'Enrolled', 'Graduate'][cls_idx]}: {count} samples")

    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    # Use zero_division=0 to handle cases with no predicted/true samples for a class in precision/recall calculation
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_labels, average='weighted', zero_division=0
    )
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted): {recall:.3f}")
    print(f"F1-Score (weighted): {f1:.3f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dropout', 'Enrolled', 'Graduate'],
                yticklabels=['Dropout', 'Enrolled', 'Graduate'])
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    return accuracy, precision, recall, f1

# Analyze class distribution in a dataset
def analyze_class_distribution(y_data, dataset_name):
    """
    Prints the distribution of classes in the given target data.
    y_data: Target labels
    dataset_name: Name of the dataset (e.g., "Training Set")
    """
    class_counts = pd.Series(y_data).value_counts().sort_index()
    total_samples = len(y_data)
    print(f"\nClass distribution in {dataset_name}:")
    for cls_idx, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class {['Dropout', 'Enrolled', 'Graduate'][cls_idx]}: {count} samples ({percentage:.2f}%)")

# Perform final evaluation on the test set
y_pred_final = fuzzy_inference(best_rules, X_test_fuzzy) # Rerun inference if needed, or use y_pred
evaluate_model(y_test, y_pred_final, title="Final Model on Test Set")

# Analyze class distributions in training and test sets
analyze_class_distribution(y_train, "Training Set")
analyze_class_distribution(y_test, "Test Set")

# Re-plot membership functions for clarity (optional, if already done and shown)
print("\nRe-plotting Membership Functions for Continuous Features:")
for feat in cont_features:
    if feat in df.columns: # Ensure the feature exists in the original dataframe for min/max/mean
        x_plot_vals = np.linspace(df[feat].min(), df[feat].max(), 1000)
        low_membership_plot = triangular_membership(x_plot_vals, df[feat].min(), df[feat].min(), df[feat].mean())
        medium_membership_plot = triangular_membership(x_plot_vals, df[feat].min(), df[feat].mean(), df[feat].max())
        high_membership_plot = triangular_membership(x_plot_vals, df[feat].mean(), df[feat].max(), df[feat].max())

        plt.figure(figsize=(8, 5))
        plt.plot(x_plot_vals, low_membership_plot, label='Low', color='blue')
        plt.plot(x_plot_vals, medium_membership_plot, label='Medium', color='green')
        plt.plot(x_plot_vals, high_membership_plot, label='High', color='red')
        plt.title(f'Triangular Membership Functions for {feat}')
        plt.xlabel(feat)
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Compute the activation degree of a rule for a given sample
def compute_rule_activation(sample_fuzzified, rule_antecedent):
    """
    Calculates how strongly a rule's antecedent is activated by a sample.
    sample_fuzzified: Tuple of fuzzified feature values for the sample
    rule_antecedent: Tuple of fuzzified feature values in the rule's IF part
    """
    degree = 1.0
    for j, feature_value_in_sample in enumerate(sample_fuzzified):
        # Similarity between sample's j-th feature and rule's j-th condition value
        similarity = 1 - abs(rule_antecedent[j] - feature_value_in_sample)
        degree *= max(0, similarity) # Aggregate using product (can also be min)
    return degree

# Show rule activations for some test samples
print("\nRule Activations for Test Samples:")
num_samples_to_show = min(5, len(X_test_fuzzy)) # Show for up to 5 samples
# Get predictions for these samples if not already available for all X_test_fuzzy
predictions_for_samples = fuzzy_inference(best_rules, X_test_fuzzy.iloc[:num_samples_to_show])


for i in range(num_samples_to_show):
    current_sample_fuzzified = tuple(X_test_fuzzy.iloc[i])
    true_label_for_sample = y_test.iloc[i]
    predicted_label_for_sample = predictions_for_samples[i] # Use pre-calculated predictions

    print(f"\nSample {i + 1} (True Class: {['Dropout', 'Enrolled', 'Graduate'][true_label_for_sample]}, "
          f"Predicted Class: {['Dropout', 'Enrolled', 'Graduate'][predicted_label_for_sample]})")

    activations = []
    # Check activation for each rule in the best rule set
    for rule_index, (antecedent, rule_label, rule_confidence) in enumerate(best_rules):
        activation_degree = compute_rule_activation(current_sample_fuzzified, antecedent)
        if activation_degree > 0: # Only consider rules that are activated
            activations.append((rule_index, activation_degree, rule_label, rule_confidence))

    # Sort activated rules by their activation degree (highest first)
    activations.sort(key=lambda x: x[1], reverse=True)

    print("Activated Rules (Top 3):")
    for act_info in activations[:3]: # Show top 3 activated rules
        rule_idx, degree, r_label, r_conf = act_info
        print(f"  Rule {rule_idx + 1}: Activation Degree = {degree:.3f}, "
              f"Class = {['Dropout', 'Enrolled', 'Graduate'][r_label]}, "
              f"Confidence = {r_conf:.2f}")
        # Optionally print the antecedent of the activated rule for better understanding
        # print("  Antecedent:")
        # for k, val_cond in enumerate(best_rules[rule_idx][0]):
        #     print(f"    {X_test_fuzzy.columns[k]} = {val_cond:.3f}")


# Interpret some of the key rules from the final optimized set
print("\nInterpretation of Key Rules:")
# Sort rules by confidence or another importance metric if desired. Here, just taking top N from best_rules.
top_n_rules_for_interpretation = sorted(best_rules, key=lambda x: x[2], reverse=True)[:3] # Top 3 by confidence

for i, (antecedent, rule_class_label, rule_conf) in enumerate(top_n_rules_for_interpretation):
    print(f"\nRule {i + 1} (Class: {['Dropout', 'Enrolled', 'Graduate'][rule_class_label]}, Confidence: {rule_conf:.2f}):")
    print("Antecedent (IF part):")
    for j, val_condition in enumerate(antecedent):
        # Show the condition for each fuzzy feature involved in the rule
        print(f"  {X_test_fuzzy.columns[j]} = {val_condition:.3f}") # Use X_test_fuzzy.columns as a proxy for feature names
    print("Interpretation:")
    if rule_class_label == 0: # Dropout
        print("This rule suggests features/conditions that increase the likelihood of a student dropping out.")
    elif rule_class_label == 1: # Enrolled
        print("This rule suggests features/conditions indicative of a student continuing their enrollment.")
    else: # Graduate
        print("This rule suggests features/conditions that increase the likelihood of a student graduating.")