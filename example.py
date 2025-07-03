import xtorch_bridge
import numpy as np

def main():
    # Example: 2x3 matrix * 3x2 matrix
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2x3 matrix
    b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]  # 3x2 matrix
    rows_a, cols_a, cols_b = 2, 3, 2

    result = xtorch_bridge.matrix_multiply(a, b, rows_a, cols_a, cols_b)
    result_array = np.array(result).reshape(rows_a, cols_b)
    print("Result of matrix multiplication:")
    print(result_array)

if __name__ == "__main__":
    main()