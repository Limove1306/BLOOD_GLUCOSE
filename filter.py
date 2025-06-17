import vitaldb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,find_peaks
from numpy.linalg import norm
import neurokit2 as nk

df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # Load clinical data
df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # Load track list
df_labs = pd.read_csv('https://api.vitaldb.net/labs')  # Load lab result

# Hàm nội suy tuyến tính các giá trị NaN trong tín hiệu PPG (nếu có)
def interpolate_nans(signal):
    x = np.array(signal, dtype=float)
    if not np.isnan(x).any():
        return x  # nếu không có NaN thì trả về chính signal
    n = len(x)
    # Tìm chỉ số của các phần tử không NaN
    idx = np.arange(n)
    not_nan_mask = ~np.isnan(x)
    not_nan_idx = idx[not_nan_mask]
    not_nan_vals = x[not_nan_mask]
    if not_nan_idx.size == 0:
        return x  # tất cả đều NaN, trả về mảng gốc (hoặc có thể xử lý khác)
    # Điền giá trị cho các vị trí NaN ở đầu hoặc cuối (nếu có) bằng giá trị gần nhất
    if np.isnan(x[0]):
        first_valid = not_nan_vals[0]
        x[:not_nan_idx[0]] = first_valid
    if np.isnan(x[-1]):
        last_valid = not_nan_vals[-1]
        x[not_nan_idx[-1]+1:] = last_valid
    # Nội suy các đoạn NaN ở giữa
    nan_mask = np.isnan(x)
    x[nan_mask] = np.interp(idx[nan_mask], not_nan_idx, not_nan_vals)
    return x

def butter_bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=100, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(ppg_signal, height_threshold, distance_threshold):
    
    x = np.array(ppg_signal)
    peaks_idx = []  # danh sách chỉ số các đỉnh
    last_peak_idx = None   # chỉ số đỉnh gần nhất đã chấp nhận
    last_peak_val = None   # biên độ của đỉnh gần nhất đã chấp nhận

    # Duyệt qua tín hiệu (bỏ qua điểm đầu và cuối để tránh vượt biên)
    for i in range(1, len(x) - 1):
        # Kiểm tra điều kiện cực đại cục bộ so với hai điểm kề trước và sau
        if x[i] > x[i-1] and x[i] > x[i+1]:
            # Kiểm tra ngưỡng biên độ tối thiểu
            if x[i] < height_threshold:
                continue  # bỏ qua nếu biên độ nhỏ hơn ngưỡng yêu cầu
            # Nếu đã có đỉnh trước đó, kiểm tra khoảng cách
            if last_peak_idx is not None:
                # Nếu khoảng cách tới đỉnh trước < ngưỡng yêu cầu
                if i - last_peak_idx < distance_threshold:
                    # Nếu đỉnh này cao hơn đỉnh trước, thay thế đỉnh trước bằng đỉnh này
                    if last_peak_val is not None and x[i] > last_peak_val:
                        peaks_idx[-1] = i
                        last_peak_idx = i
                        last_peak_val = x[i]
                    # Nếu đỉnh này không cao bằng đỉnh trước, bỏ qua nó
                    continue  # chuyển sang điểm tiếp theo
            # Nếu chưa có đỉnh trước hoặc khoảng cách đã đủ lớn, chấp nhận đỉnh này
            peaks_idx.append(i)
            last_peak_idx = i
            last_peak_val = x[i]
    return np.array(peaks_idx, dtype=int)

def count_peaks_in_window(window, height_threshold, distance_threshold):
    peaks = detect_peaks(window, height_threshold, distance_threshold)
    return len(peaks)


def extract_windows(ppg_signal, peaks, window_size, height_threshold, distance_thresho ld):
    x = np.array(ppg_signal)
    windows = []
    half_window = window_size // 2

    for p in peaks:
        left_idx = p - half_window
        right_idx = p + half_window
        if left_idx < 0 or right_idx > len(x):
            continue
        window = x[left_idx:right_idx]

        if len(window) == window_size and count_peaks_in_window(window, height_threshold, distance_threshold) == 1:
            windows.append(window)
    return np.array(windows)

def compute_template(windows):

    if len(windows) == 0:
        return None
    W = np.array(windows)
    # Tính trung bình theo cột (mỗi vị trí thời gian trong cửa sổ)
    template = np.mean(W, axis=0)
    return template

def cosine_similarity(window, template):

    # Chuyển về numpy array phòng trường hợp đầu vào không phải numpy
    a = np.array(window, dtype=float)
    b = np.array(template, dtype=float)
    # Đảm bảo hai vector có cùng độ dài
    if a.shape != b.shape:
        raise ValueError("Cửa sổ và mẫu chuẩn phải có cùng độ dài")
    # Tính tích vô hướng và chuẩn (norm) của hai vector
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # nếu vector phẳng (norm=0) thì cho độ tương đồng = 0
    cos_sim = dot / (norm_a * norm_b)
    return cos_sim

def filter_windows_by_similarity(windows, template, similarity_threshold):

    filtered_windows = []
    for w in windows:
        sim = cosine_similarity(w, template)
        if sim >= similarity_threshold:
            filtered_windows.append(w)
    return np.array(filtered_windows)
    
def main(ppg_signal, height_threshold, distance_threshold,window_size):
    if np.isnan(ppg_signal).all() or len(ppg_signal) == 0:
        print("Tín hiệu toàn NaN hoặc rỗng, bỏ qua.")
        return np.array([])

    # 1. Nội suy NaN
    signal = interpolate_nans(ppg_signal)

    # 2. Lọc Butterworth
    signal = butter_bandpass_filter(signal)

    # 3. Phát hiện đỉnh
    peaks = detect_peaks(signal, height_threshold, distance_threshold)

    # 4. Trích xuất cửa sổ quanh mỗi đỉnh
    windows = extract_windows(signal, peaks, window_size, height_threshold, distance_threshold)

    # 5. Tính template và lọc theo cosine
    template = compute_template(windows)
    if template is None:
        return np.array([])

    high_quality_windows = filter_windows_by_similarity(windows, template, similarity_threshold)
    return high_quality_windows

    # Tạo thư mục chứa segment nếu chưa tồn tại
os.makedirs("segments_2", exist_ok=True)

caseids = list(
    set(df_trks.loc[df_trks['tname'] == 'SNUADC/PLETH', 'caseid']) &
    set(df_labs.loc[df_labs['name'] == 'gluc', 'caseid'])
)
#caseids = caseids[:20]
# Tham số mặc định
fs = 100                        # Tần số lấy mẫu (Hz)
segment_duration = 10 * fs
window_size = int(1 * fs)       # Kích thước cửa sổ (số mẫu trong 1 giây)
height_threshold = 20          # Ngưỡng biên độ tối thiểu để nhận dạng đỉnh (có thể điều chỉnh)
distance_threshold = 0.8 * fs         # Khoảng cách tối thiểu giữa hai đỉnh (theo số mẫu)
similarity_threshold = 0.85
total = 0
print(len(caseids))

# Danh sách để lưu tên file và nhãn tương ứng
labels_data = []
total = 0
for caseid in caseids:
    print(f"loading {caseid}...", flush=True)

    # Lấy tất cả các lần đo glucose hợp lệ (dt >= 0)
    gluc_rows = df_labs[(df_labs['caseid'] == caseid) & 
                        (df_labs['name'] == 'gluc') & 
                        (df_labs['dt'] >= 0)].sort_values(by='dt')
    
    if len(gluc_rows) == 0:
        continue

    # Load tín hiệu PPG 1 lần để tái sử dụng
    ppg = vitaldb.load_case(caseid, ['SNUADC/PLETH'], 1/fs)[:, 0]

    segment_count = 1  # Reset số thứ tự segment cho mỗi caseid

    for _, row in gluc_rows.iterrows():
        tm = row['dt']
        gluc_value = row['result']

        center_idx = int(tm * fs)
        start = max(center_idx - 48000, 0)
        end = min(center_idx + 48000, len(ppg))
        ppg_segment = ppg[start:end]

        if len(ppg_segment) < 96000:
            continue

        # Chia đoạn 16 phút thành các segment 10s
        for i in range(0, len(ppg_segment) - segment_duration + 1, segment_duration):
            sub_segment = ppg_segment[i:i+segment_duration]
            if np.isnan(sub_segment).all():
                continue

            if len(sub_segment) <= 21:
                continue

            try:
                windows = main(sub_segment, height_threshold, distance_threshold, window_size)
                total += len(windows)

            except Exception as e:
                print(f"Lỗi xử lý đoạn nhỏ: {e}")
                continue

            for window in windows:
                filename = f"{caseid}_{segment_count:05d}.csv"
                filepath = os.path.join("segments_2", filename)

                pd.DataFrame(window).to_csv(filepath, index=False, header=False)
                labels_data.append([filename, gluc_value])
                segment_count += 1

    print(f"Total: {total}")
    
# Lưu file nhãn
labels_df = pd.DataFrame(labels_data, columns=["filename", "gluc"])
labels_df.to_csv("labels_2.csv", index=False)

print(f"\n Tổng cửa sổ chất lượng cao thu được: {total}")
print(f"Chênh lệch so với bài báo (699072): {699072 - total}")
print(f"\n Đã lưu {len(labels_df)} segments vào thư mục segments/")