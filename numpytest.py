import numpy as np

arr = np.arange(3)
items_to_delete = np.where(arr == 2)
print(items_to_delete)
new_arr = np.delete(arr, items_to_delete)
print(new_arr)
