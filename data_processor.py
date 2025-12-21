import pandas as pd  # 匯入 pandas 套件，用於處理資料表格
import numpy as np  # 匯入 numpy 套件，用於數值運算
from sklearn.model_selection import train_test_split  # 從 sklearn 匯入資料切割工具
import os  # 匯入 os 套件，用於檔案操作
import joblib  # 匯入 joblib 套件，用於儲存模型或資料

DATA_PATH = 'kaohsiung_main.csv'  # 定義資料檔案的路徑

def clean_roc_year(roc_date):
    # 將民國年份轉換為西元年份
    try:
        if pd.isna(roc_date): return None  # 如果日期是空值，回傳 None
        date_str = str(int(roc_date))  # 將日期轉為字串
        if len(date_str) < 6: return None  # 如果字串長度小於 6，回傳 None
        year = int(date_str[:-4]) + 1911  # 提取年份並加上 1911
        return year  # 回傳轉換後的年份
    except:
        return None  # 如果發生錯誤，回傳 None

def parse_building_type(text):
    # 將建物型態文字轉換為數值編碼
    if pd.isna(text): return 0  # 如果型態是空值，回傳 0
    text = str(text)  # 將型態轉為字串
    if '大樓' in text or '華廈' in text: return 3  # 如果包含「大樓」或「華廈」，回傳 3
    elif '透天' in text or '別墅' in text: return 2  # 如果包含「透天」或「別墅」，回傳 2
    elif '公寓' in text: return 1  # 如果包含「公寓」，回傳 1
    else: return 0  # 其他情況回傳 0

def load_and_clean_data():
    # 載入並清理資料
    if not os.path.exists(DATA_PATH):  # 檢查資料檔案是否存在
        print(f"❌ 錯誤：找不到 {DATA_PATH}")  # 如果不存在，印出錯誤訊息
        return None  # 回傳 None

    print(f"🧹 啟動 V3 嚴格資料清洗: {DATA_PATH} ...")  # 印出清洗資料的訊息

    try:
        df = pd.read_csv(DATA_PATH, header=1)  # 嘗試讀取資料，跳過第一列
    except:
        df = pd.read_csv(DATA_PATH)  # 如果失敗，重新讀取資料

    # 防呆：確認欄位
    if '主要用途' not in df.columns:  # 如果資料中沒有「主要用途」欄位
        df = pd.read_csv(DATA_PATH)  # 重新讀取資料

    initial_count = len(df)  # 記錄原始資料筆數

    # 1. 【關鍵】只保留「房地」交易 (排除單獨買土地、車位)
    # 這是之前分數低的主因！
    if '交易標的' in df.columns:
        df = df[df['交易標的'].str.contains('房地', na=False)]  # 篩選包含「房地」的交易

    # 2. 篩選「住家用」
    df = df[df['主要用途'] == '住家用']  # 只保留主要用途為「住家用」的資料
    
    # 3. 排除「特殊交易」 (如親友交易、急買急賣)
    # 備註欄如果有寫字，通常就是特殊交易，價格不準
    if '備註' in df.columns:
        # 只保留備註是空的，或者內容很少的
        df = df[df['備註'].isna()]  # 篩選備註為空值的資料

    # 4. 處理年份
    df['trade_year'] = df['交易年月日'].apply(clean_roc_year)  # 將交易日期轉換為年份
    df['build_year'] = df['建築完成年月'].apply(clean_roc_year)  # 將建築完成日期轉換為年份
    df['house_age'] = df['trade_year'] - df['build_year']  # 計算屋齡
    
    # 🌟 擴大資料範圍：改抓 2021 (民國110年) 之後
    # 讓模型吃飽一點，但不要太舊
    df = df[df['trade_year'] >= 2021]  # 只保留 2021 年以後的資料

    # 5. 處理價格與坪數
    df['price_per_ping'] = (pd.to_numeric(df['單價元平方公尺'], errors='coerce') / 0.3025) / 10000  # 計算每坪價格
    df['total_ping'] = pd.to_numeric(df['建物移轉總面積平方公尺'], errors='coerce') * 0.3025  # 計算總坪數

    # 6. 樓層處理
    def parse_floor(x):
        # 將樓層文字轉換為數值
        try:
            if pd.isna(x): return 1  # 如果樓層是空值，回傳 1
            txt = str(x).replace('層', '')  # 移除「層」字
            return float(txt) if txt.isdigit() else 1  # 如果是數字，轉為浮點數，否則回傳 1
        except:
            return 1  # 如果發生錯誤，回傳 1
    df['total_floors'] = df['總樓層數'].apply(parse_floor)  # 處理總樓層數

    # 🌟 排除「一樓」 (店面效應)
    if '移轉層次' in df.columns:
        # 如果移轉層次包含 "一層" 或 "1層"，通常是店面，價格極高會干擾
        df = df[~df['移轉層次'].astype(str).str.contains('一', na=False)]  # 排除包含「一層」的資料
        df = df[~df['移轉層次'].astype(str).str.contains('1', na=False)]  # 排除包含「1層」的資料

    # 7. 地點編碼
    df['district_code'] = df['鄉鎮市區'].astype('category').cat.codes  # 將地區轉換為數值編碼

    # 8. 建物型態
    type_col = '建物型態'  # 預設建物型態欄位名稱
    if type_col not in df.columns:  # 如果欄位不存在
        for col in df.columns:  # 遍歷所有欄位
            if '型態' in col: type_col = col; break  # 找到包含「型態」的欄位
    df['building_type'] = df[type_col].apply(parse_building_type)  # 將建物型態轉換為數值

    # 9. 最後清洗
    df.dropna(subset=['price_per_ping', 'house_age', 'total_ping'], inplace=True)  # 移除空值資料
    df = df[df['price_per_ping'] > 5]  # 排除每坪價格過低的資料
    df = df[df['price_per_ping'] < 150]  # 收緊上限，排除極端豪宅
    df = df[df['house_age'] >= 0]  # 排除預售屋還沒蓋好屋齡負的
    df = df[df['house_age'] < 60]  # 排除極老屋

    print(f"✨ V3 清洗完成！")  # 印出清洗完成訊息
    print(f"   從 {initial_count} 筆 -> 精選出 {len(df)} 筆「高品質」純住宅資料")  # 印出清洗後的資料筆數

    return df  # 回傳清洗後的資料

def prepare_and_split():
    # 準備並切割資料
    df = load_and_clean_data()  # 載入並清理資料
    if df is None or len(df) == 0: return None, None, None, None  # 如果資料為空，回傳 None

    # 加入 trade_year 讓模型知道年份的通膨影響
    features = ['house_age', 'total_ping', 'total_floors', 'district_code', 'building_type', 'trade_year']  # 定義特徵欄位
    target = 'price_per_ping'  # 定義目標欄位

    X = df[features]  # 提取特徵資料
    y = df[target]  # 提取目標資料

    joblib.dump(X.columns.tolist(), 'feature_names.pkl')  # 儲存特徵名稱

    # 切割資料
    return train_test_split(X, y, test_size=0.2, random_state=42)  # 將資料切割為訓練集與測試集

if __name__ == "__main__":
    prepare_and_split()  # 如果是主程式，執行資料準備與切割