import numpy as np


def load_csv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    
    # Đọc dữ liệu (bỏ header)
    data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    
    return data, header


def missing_rate(col):
    return np.mean((col == '') | (col == 'NaN') | (col == 'nan'))


def get_missing_stats(data, header):
    missing_stats = {}
    for i, colname in enumerate(header):
        rate = missing_rate(data[:, i])
        missing_stats[colname] = rate
    
    return missing_stats


def process_experience_column(col):
    col_processed = col.copy()
    
    # Chuyển '>20' thành 21 (giả sử lớn hơn 20)
    col_processed[col_processed == '>20'] = '21'
    
    # Chuyển '<1' thành 0.5 (giả sử ít hơn 1)
    col_processed[col_processed == '<1'] = '0.5'
    
    # Chuyển 'never' thành 0
    col_processed[col_processed == 'never'] = '0'
    
    # Thay giá trị rỗng thành 'nan'
    col_processed[col_processed == ''] = 'nan'
    
    return col_processed


def get_column_by_name(data, header, colname):
    idx = header.index(colname)
    return data[:, idx]


def get_numeric_column(data, header, colname, process_experience=False):
    """
    Lấy cột numeric và chuyển sang float
    """
    col = get_column_by_name(data, header, colname)
    col_processed = col.copy()
    
    if process_experience and colname == 'experience':
        col_processed = process_experience_column(col_processed)
    else:
        # Thay giá trị rỗng thành 'nan'
        col_processed[col_processed == ''] = 'nan'
    
    # Chuyển sang float
    col_float = col_processed.astype(float)
    
    return col_float


def get_target(data, header):
    """
    Lấy cột target và chuyển sang int
    """
    target_col = get_column_by_name(data, header, 'target')
    target = target_col.astype(float).astype(int)
    return target


def split_by_target(data_col, target_col):
    data_target_0 = data_col[target_col == 0]
    data_target_1 = data_col[target_col == 1]
    return data_target_0, data_target_1


def get_categorical_stats(data, header, colname):
    col = get_column_by_name(data, header, colname)
    
    # Loại bỏ giá trị missing
    col_clean = col[col != '']
    
    # Đếm tần suất
    unique_vals, counts = np.unique(col_clean, return_counts=True)
    
    # Sắp xếp theo tần suất giảm dần
    sort_idx = np.argsort(-counts)
    unique_vals = unique_vals[sort_idx]
    counts = counts[sort_idx]
    
    missing_count = np.sum(col == '')
    missing_rate_val = missing_count / len(col)
    
    return {
        'unique_vals': unique_vals,
        'counts': counts,
        'missing_count': missing_count,
        'missing_rate': missing_rate_val
    }


# preprocessing pipeline 

def one_hot_encode(data, header, colname, fill_missing='Missing'):
    col = get_column_by_name(data, header, colname)
    col_filled = col.copy()
    col_filled[col_filled == ''] = fill_missing
    categories = np.unique(col_filled)
    n_samples = len(col_filled)
    encoded = np.zeros((n_samples, len(categories)), dtype=int)
    for i, category in enumerate(categories):
        encoded[col_filled == category, i] = 1
    return encoded, categories.tolist()


def label_encode(data, header, colname):
    col = get_column_by_name(data, header, colname)
    unique_vals = np.unique(col)
    label_map = {val: i for i, val in enumerate(unique_vals)}
    encoded = np.array([label_map[val] for val in col], dtype=int)
    return encoded, label_map


def normalize_features(X, method='standard'):
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        X_norm = (X - mean) / std
        return X_norm, {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        X_norm = (X - min_vals) / range_vals
        return X_norm, {'min': min_vals, 'max': max_vals}
    else:
        raise ValueError("Unsupported normalization method")


def add_bias(X):
    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1))
    return np.hstack([bias, X])


def build_feature_matrix(data, header):
    # Numeric columns
    cdi = get_numeric_column(data, header, 'city_development_index')
    training_hours = get_numeric_column(data, header, 'training_hours')
    experience = get_numeric_column(data, header, 'experience', process_experience=True)
    experience_median = np.nanmedian(experience)
    experience_filled = experience.copy()
    experience_filled[np.isnan(experience_filled)] = experience_median

    # Categorical columns (selected)
    categorical_cols = [
        'relevent_experience', 'enrolled_university', 'education_level',
        'last_new_job', 'company_size'
    ]
    encoded_features = {}
    all_categories = {}
    for colname in categorical_cols:
        encoded, categories = one_hot_encode(data, header, colname)
        encoded_features[colname] = encoded
        all_categories[colname] = categories

    # City label encode
    city_encoded, city_label_map = label_encode(data, header, 'city')

    feature_list = []
    # Base numeric
    feature_list.append(cdi.reshape(-1, 1))
    feature_list.append(training_hours.reshape(-1, 1))
    feature_list.append(experience_filled.reshape(-1, 1))
    feature_list.append(city_encoded.reshape(-1, 1))

    # Interaction and engineered features
    cdi_exp_interaction = (cdi * experience_filled).reshape(-1, 1)
    feature_list.append(cdi_exp_interaction)
    cdi_squared = (cdi ** 2).reshape(-1, 1)
    feature_list.append(cdi_squared)
    exp_squared = (experience_filled ** 2).reshape(-1, 1)
    feature_list.append(exp_squared)
    training_per_exp = (training_hours / (experience_filled + 1)).reshape(-1, 1)
    feature_list.append(training_per_exp)
    cdi_per_exp = (cdi / (experience_filled + 1)).reshape(-1, 1)
    feature_list.append(cdi_per_exp)
    exp_log = np.log1p(experience_filled).reshape(-1, 1)
    feature_list.append(exp_log)
    training_log = np.log1p(training_hours).reshape(-1, 1)
    feature_list.append(training_log)

    # Categorical one-hot
    for colname in categorical_cols:
        feature_list.append(encoded_features[colname])

    X = np.hstack(feature_list)
    y = get_target(data, header)

    meta = {
        'categorical_cols': categorical_cols,
        'all_categories': all_categories,
        'city_label_map': city_label_map,
        'experience_median': experience_median,
        'n_base_features': 4,
        'n_engineered_features': X.shape[1] - 4 - sum(encoded_features[c].shape[1] for c in categorical_cols),
        'n_categorical_features': sum(encoded_features[c].shape[1] for c in categorical_cols)
    }
    return X, y, meta


def stratified_split_save(X, y, processed_dir, test_size=0.2, seed=42):
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(class0_idx)
    rng.shuffle(class1_idx)
    n_class0_test = int(len(class0_idx) * test_size)
    n_class1_test = int(len(class1_idx) * test_size)
    test_idx = np.concatenate([class0_idx[:n_class0_test], class1_idx[:n_class1_test]])
    train_idx = np.concatenate([class0_idx[n_class0_test:], class1_idx[n_class1_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    import os
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_idx': train_idx,
        'test_idx': test_idx
    }


def stratified_split(X, y, test_size=0.2, seed=42):
    """Stratified split without saving to disk."""
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(class0_idx)
    rng.shuffle(class1_idx)
    n_class0_test = int(len(class0_idx) * test_size)
    n_class1_test = int(len(class1_idx) * test_size)
    test_idx = np.concatenate([class0_idx[:n_class0_test], class1_idx[:n_class1_test]])
    train_idx = np.concatenate([class0_idx[n_class0_test:], class1_idx[n_class1_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx]
    }


def save_processed(processed_dir, X_train, y_train, X_test, y_test):
    import os
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    return True


def preprocess_train(csv_path, processed_dir, test_size=0.2, seed=42):
    """Full preprocessing pipeline for training CSV.

    Returns a dictionary with matrices, metadata and split info.
    """
    data, header = load_csv_data(csv_path)
    X_raw, y, meta = build_feature_matrix(data, header)
    X_norm, norm_stats = normalize_features(X_raw, method='standard')
    X_final = add_bias(X_norm)
    split_info = stratified_split_save(X_final, y, processed_dir, test_size=test_size, seed=seed)
    result = {
        'X_raw': X_raw,
        'X_final': X_final,
        'y': y,
        'meta': meta,
        'norm_stats': norm_stats,
        'split': split_info,
        'header': header
    }
    return result

