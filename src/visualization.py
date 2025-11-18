"""
Module visualization cho project HR Job Change
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_target_distribution(target, save_path=None):
    """
    Vẽ biểu đồ phân phối target
    
    Parameters:
    -----------
    target : np.ndarray
        Cột target
    save_path : str, optional
        Đường dẫn lưu file (nếu None thì chỉ hiển thị)
    """
    unique_vals, counts = np.unique(target, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(unique_vals, counts, color=['#007acc', '#ff6347'])
    plt.xticks(unique_vals, ['Không đổi việc (0)', 'Đổi việc (1)'])
    plt.ylabel("Số lượng")
    plt.title("Phân phối biến mục tiêu (target)")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_histogram(data, title, xlabel, bins=30, color='#0099cc', alpha=0.7, save_path=None):
    """
    Vẽ histogram
    
    Parameters:
    -----------
    data : np.ndarray
        Dữ liệu (loại bỏ nan trước khi truyền vào)
    title : str
        Tiêu đề biểu đồ
    xlabel : str
        Nhãn trục x
    bins : int
        Số bins
    color : str
        Màu sắc
    alpha : float
        Độ trong suốt
    save_path : str, optional
        Đường dẫn lưu file
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Tần suất")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_boxplot(data, labels=None, ylabel='', title='', save_path=None):
    """
    Vẽ boxplot
    
    Parameters:
    -----------
    data : list of np.ndarray or np.ndarray
        Dữ liệu (nếu list thì vẽ nhiều boxplot)
    labels : list, optional
        Nhãn cho các boxplot
    ylabel : str
        Nhãn trục y
    title : str
        Tiêu đề
    save_path : str, optional
        Đường dẫn lưu file
    """
    plt.figure(figsize=(8, 6))
    
    if isinstance(data, list):
        plt.boxplot(data, labels=labels, vert=True)
    else:
        plt.boxplot(data, vert=True)
    
    plt.ylabel(ylabel)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_categorical_barh(unique_vals, counts, colname, top_n=10, save_path=None):
    """
    Vẽ bar chart ngang cho cột categorical
    
    Parameters:
    -----------
    unique_vals : np.ndarray
        Các giá trị unique (đã sắp xếp theo tần suất giảm dần)
    counts : np.ndarray
        Số lượng của mỗi giá trị
    colname : str
        Tên cột
    top_n : int
        Số giá trị top cần hiển thị
    save_path : str, optional
        Đường dẫn lưu file
    """
    top_n = min(top_n, len(unique_vals))
    
    plt.figure(figsize=(10, 6))
    plt.barh(unique_vals[:top_n][::-1], counts[:top_n][::-1], 
             color='#ff9999', alpha=0.7)
    plt.xlabel("Số lượng")
    plt.title(f"Phân phối {colname} (top {top_n})")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_comparison_histogram(data_0, data_1, label_0, label_1, 
                              xlabel, title, bins=30, save_path=None):
    """
    Vẽ histogram so sánh giữa 2 nhóm
    
    Parameters:
    -----------
    data_0 : np.ndarray
        Dữ liệu nhóm 0
    data_1 : np.ndarray
        Dữ liệu nhóm 1
    label_0 : str
        Nhãn nhóm 0
    label_1 : str
        Nhãn nhóm 1
    xlabel : str
        Nhãn trục x
    title : str
        Tiêu đề
    bins : int
        Số bins
    save_path : str, optional
        Đường dẫn lưu file
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data_0, bins=bins, alpha=0.5, label=label_0, color='#007acc')
    plt.hist(data_1, bins=bins, alpha=0.5, label=label_1, color='#ff6347')
    plt.xlabel(xlabel)
    plt.ylabel("Tần suất")
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_comparison_boxplot(data_0, data_1, label_0, label_1, 
                            ylabel, title, save_path=None):
    """
    Vẽ boxplot so sánh giữa 2 nhóm
    
    Parameters:
    -----------
    data_0 : np.ndarray
        Dữ liệu nhóm 0
    data_1 : np.ndarray
        Dữ liệu nhóm 1
    label_0 : str
        Nhãn nhóm 0
    label_1 : str
        Nhãn nhóm 1
    ylabel : str
        Nhãn trục y
    title : str
        Tiêu đề
    save_path : str, optional
        Đường dẫn lưu file
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot([data_0, data_1], labels=[label_0, label_1])
    plt.ylabel(ylabel)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

