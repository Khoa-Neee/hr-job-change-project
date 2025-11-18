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
