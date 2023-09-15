import numpy as np
from sklearn.model_selection import KFold


#np.random.seed(42) #seed set to 42 for reproducibility... this can be changed and it will reliably show that car sales have the most impact on ebike sales.

# Defining the input data
sales_data = np.array([130, 152, 263, 369, 423, 416, 750, 928])
battery_prices = np.array([381, 293, 219, 180, 156, 135, 132, 135])
disposable_income = np.array([41383, 41821, 42699, 43886, 44644, 47241, 48219, 0])
car_sales = np.array([17408, 17477.3, 17150.1, 17224.9, 16961.1, 14471.8, 14926.9, 13700])


# Combining the input data into a single array
input_data = np.column_stack((battery_prices, disposable_income, car_sales))

# Defining the output data
output_data = sales_data.reshape((-1, 1))



# Define the neural network architecture
weights1 = np.random.rand(3, 8) * 0.1
weights2 = np.random.rand(8, 1) * 0.1

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def forward(input_data):
    global weights1, weights2
    layer1 = relu(np.dot(input_data, weights1))
    output = linear(np.dot(layer1, weights2))
    return output

def backward(input_data, output_data, learning_rate):
    global weights1, weights2
    layer1 = relu(np.dot(input_data, weights1))
    output = linear(np.dot(layer1, weights2))
    error = output - output_data
    d_weights2 = np.dot(layer1.T, error)
    d_layer1 = np.dot(error, weights2.T)
    d_layer1[layer1 <= 0] = 0
    d_weights1 = np.dot(input_data.T, d_layer1)
    weights2 -= learning_rate * d_weights2
    weights1 -= learning_rate * d_weights1

# Define the number of folds for cross-validation
n_folds = 5

# Define the learning rate and number of epochs for training
learning_rate = 0.0001
epochs = 1000000

# Create a KFold object to split the data into folds
kf = KFold(n_splits=n_folds, shuffle=True)

# Initialize a list to store the performance of the model on each fold
fold_scores = []

# Loop over the folds
for train_indices, test_indices in kf.split(input_data):
    
    # Split the data into training and testing sets
    X_train, X_test = input_data[train_indices], input_data[test_indices]
    y_train, y_test = output_data[train_indices], output_data[test_indices]
    
    # Train the model
    for i in range(epochs):
        backward(X_train, y_train, learning_rate)
    
    # Evaluate the model on the testing set
    y_pred = forward(X_test)
    score = np.mean(np.abs(y_pred - y_test))
    
    # Store the performance of the model on this fold
    fold_scores.append(score)

# Calculate the mean and standard deviation of the performance scores across folds
mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

# Print the cross-validation results
print("Cross-validation results:")
print("Mean score:", mean_score)
print("Standard deviation:", std_score)
#Calculate the variable importance

importance = np.abs(weights1).sum(axis=1) + np.abs(weights2).sum(axis=0)
sorted_indices = np.argsort(importance)[::-1]

#Print the variable importance in order

# Define the names of the input variables
# Define the names of the input variables
input_names = ['Battery Prices', 'Disposable Income', 'Car Sales']


# Print the variable importance in order
total_importance = np.sum(importance)
print("\nVariable importance in order:")
for i in sorted_indices:
    print(input_names[sorted_indices[i]] + ": " + str(importance[i]))
    importance_percentage = importance[i] / total_importance * 100
    print(input_names[sorted_indices[i]] + " weight percent: " + str(importance_percentage) + "%")


def evolve(self):
    parent1, parent2 = self.select()
    child = self.crossover(parent1, parent2)
    self.mutate(child)
    return child

#Define the main loop of the simulation

gen_alg = GeneticAlgorithm(population_size, 4, 4) # 4 inputs (food x, food y, circle x, circle y), 4 outputs (direction)
clock = pygame.time.Clock()
while True:
# Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()



# Update the circles' positions based on the neural network's output
for circle in population:
    inputs = [food_pos[0], food_pos[1], circle.pos_x, circle.pos_y]
    outputs = gen_alg.population[0].predict(inputs) # Use the first neural network of the population
    direction = outputs.index(max(outputs))
    circle.move(direction)
    circle.draw()

# Check if a circle has reached the food
for circle in population:
    distance = math.sqrt((circle.pos_x - food_pos[0])**2 + (circle.pos_y - food_pos[1])**2)
    if distance <= circle.radius:
        food_pos = (random.randint(0, window_size[0]), random.randint(0, window_size[1]))
        gen_alg.fitness[population.index(circle)] += 1

# Evolve the population
gen_alg.evolve()
gen_alg.population[0] = gen_alg.select()[0] # Replace the least fit individual with the new one

# Draw the food
pygame.draw.circle(window, (0, 255, 0), food_pos, 10)

# Update the window
pygame.display.update()

# Limit the frame rate
clock.tick(60)


