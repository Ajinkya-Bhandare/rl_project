import numpy as np

def flood_fill(array, start_point, new_value):
    """
    Performs a 4-way flood fill on a NumPy array.

    Args:
        array (np.ndarray): The input array.
        start_point (tuple): The (row, column) starting point for the fill.
        new_value: The value to fill the connected region with.
    """
    
    rows, cols = array.shape
    start_value = array[start_point]
    
    if start_value == new_value:
        return  # Avoid infinite recursion if start_value is already new_value

    stack = [start_point]
    
    while stack:
        row, col = stack.pop()
        if 0 <= row < rows and 0 <= col < cols and array[row, col] == start_value:
            array[row, col] = new_value            
            stack.append((row + 1, col))  # Down
            stack.append((row - 1, col))  # Up
            stack.append((row, col + 1))  # Right
            stack.append((row, col - 1))  # Left

    # print(array, new_value)
    return array