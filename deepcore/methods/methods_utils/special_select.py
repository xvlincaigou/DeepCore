import numpy as np

def check(threshold, idx, connection, distance, num_selected):
    visited = set()
    indices = []
    for i in idx:
        if i in visited:
            continue
        visited.add(i)
        indices.append(i)
        if len(indices) >= num_selected:
            break
        _, col_indices = connection[i].nonzero()
        for j in col_indices:
            if distance[i, j] < threshold:
                visited.add(j)
    return np.array(indices)

def special_select(idx, num_selected, connection, distance):
    L, R = distance.min(), distance.max()
    ans = R
    eps = 1e-9
    while abs(R - L) > eps:
        print(f'L = {L}, R = {R}')
        mid = (L + R) / 2
        indices = check(mid, idx, connection, distance, num_selected)
        if len(indices) < num_selected:
            R = mid
            ans = mid
        else:
            L = mid
    print(f'Threshold = {ans}')
    selected = indices
    return selected
