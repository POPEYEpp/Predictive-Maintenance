import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import os

# ==========================================
# 1. ฟังก์ชันตั้งค่าเกณฑ์ ISO 10816-3
# ==========================================
def get_iso_limits(machine_group='2_4', foundation='rigid'):
    """คืนค่าเกณฑ์ (mm/s RMS) ตาม ISO 10816-3"""
    # ค่าเริ่มต้นสำหรับ Group 2 & 4, Rigid Foundation
    return {'good': 1.4, 'satisfactory': 2.8, 'unsatisfactory': 4.5}

# ==========================================
# 2. ฟังก์ชันพยากรณ์ล่วงหน้า (Predictive Forecasting)
# ==========================================
def predict_trend(df, future_days=90):
    """ใช้ Linear Regression วิเคราะห์แนวโน้มจากข้อมูลในอดีต"""
    df['DateOrdinal'] = df['Time stamp'].map(pd.Timestamp.toordinal)
    X = df[['DateOrdinal']].values
    y = df['Amplitude'].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = df['Time stamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    future_pred = model.predict(future_ordinals)
    return future_dates, future_pred, model

# ==========================================
# 3. ฟังก์ชันหลักสำหรับการพล็อตและเซฟรูป
# ==========================================
def analyze_and_save_plot(file_path, machine_name, output_filename="vibration_result.png"):
    try:
        # อ่านไฟล์ (รองรับการคั่นด้วย Tab หรือ Comma)
        try:
            df = pd.read_csv(file_path, sep='\t')
            if 'Time stamp' not in df.columns:
                df = pd.read_csv(file_path, sep=',')
        except:
            df = pd.read_csv(file_path, sep=',')
            
        df['Time stamp'] = pd.to_datetime(df['Time stamp'])
        df = df.sort_values('Time stamp')
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return

    limits = get_iso_limits()
    future_dates, future_pred, model = predict_trend(df, future_days=90)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ใส่สีพื้นหลังตามโซน ISO 10816-3
    ax.axhspan(0, limits['good'], facecolor='green', alpha=0.3, label='Newly Commissioned')
    ax.axhspan(limits['good'], limits['satisfactory'], facecolor='yellow', alpha=0.3, label='Unrestricted Operation')
    ax.axhspan(limits['satisfactory'], limits['unsatisfactory'], facecolor='orange', alpha=0.3, label='Restricted Operation')
    ax.axhspan(limits['unsatisfactory'], max(df['Amplitude'].max() * 1.5, limits['unsatisfactory'] + 3), facecolor='red', alpha=0.3, label='Damage Occurs')

    # พล็อตข้อมูลจริงและเส้นพยากรณ์
    ax.plot(df['Time stamp'], df['Amplitude'], marker='o', color='blue', linewidth=2, label='Actual Amplitude')
    ax.plot(future_dates, future_pred, linestyle='--', color='red', linewidth=2, label='Predicted Trend')

    # จัดรูปแบบกราฟ
    ax.set_title(f'Vibration Trend & Prediction (ISO 10816-3) - {machine_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Stamp', fontsize=12)
    ax.set_ylabel('Amplitude (Velocity mm/s RMS)', fontsize=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 💾 บันทึกเป็นไฟล์ภาพแทนการโชว์
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ วิเคราะห์เสร็จสิ้น! บันทึกภาพผลลัพธ์ไว้ที่: {output_filename}")
    plt.close()

# ==========================================
# 4. จุดเริ่มต้นการทำงาน
# ==========================================
if __name__ == "__main__":
    # จำลองไฟล์เทสต์ หากยังไม่มีไฟล์
    test_file = 'sample_data.txt'
    if not os.path.exists(test_file):
        dates = pd.date_range(start="2024-06-01", periods=10, freq="15D")
        amps = [0.5, 0.8, 1.2, 1.5, 1.9, 2.3, 2.6, 2.9, 3.2, 3.6]
        pd.DataFrame({'Time stamp': dates, 'Amplitude': amps}).to_csv(test_file, sep='\t', index=False)
        print("สร้างไฟล์ sample_data.txt สำหรับทดสอบแล้ว")
    
    # รันโค้ดเพื่อสร้างภาพ
    analyze_and_save_plot(test_file, machine_name='CH-06 A', output_filename='sample_data.png')