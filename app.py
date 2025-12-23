import streamlit as st  #匯入 Streamlit 套件，用於建立網頁應用程式
import pandas as pd  #匯入 pandas 套件，用於處理表格資料
import joblib  #匯入joblib 套件，用於載入儲存的模型
import os  #匯入os套件，用於檢查檔案是否存在
import matplotlib.pyplot as plt#匯入 matplotlib 套件，用於繪製圖表

with st.sidebar:# 建立側邊欄
    st.header("關於本專案")#顯示專案標題
    st.markdown("學生:陳信佑。")#顯示姓名
    st.markdown("學號:4B2G0054。") #顯示學號
    st.markdown("動態程式語言期末專案。")#顯示專案名稱
    st.markdown("---")#插入分隔線
    st.info("💡 使用大數據機器學習技術，協助使用者進行房產估價與趨勢判讀。")#顯示專案資訊

#設定網頁標題與排版
st.set_page_config(#設定網頁的基本配置
    page_title="2025+ 高雄未來房價預測",#設定網頁標題
    page_icon="🔮",#設定網頁圖示
    layout="centered"#設定網頁版面為置中
)
st.caption("資料來源：內政部實價登錄平台 (2021-2025) | 模型演算法：Gradient Boosting Regressor")#顯示資料來源與模型資訊

#載入模型與資料
@st.cache_resource#快取模型資源
def load_model():
    if os.path.exists('house_price_model.pkl'):#檢查模型檔案是否存在
        return joblib.load('house_price_model.pkl')#載入模型
    return None#如果檔案不存在，回傳None

@st.cache_data#快取資料，避免重複處理
def get_district_map():
    if os.path.exists('kaohsiung_main.csv'):#檢查高雄地產csv資料檔是否存在
        try:
            df = pd.read_csv('kaohsiung_main.csv', header=1)#讀取資料，跳過第一列
        except:
            df = pd.read_csv('kaohsiung_main.csv')#失敗，重新讀取資料
        
        if '鄉鎮市區' not in df.columns: df = pd.read_csv('kaohsiung_main.csv')#確認欄位是否存在
        df['district_code'] = df['鄉鎮市區'].astype('category').cat.codes#將地區轉換為數值編碼
        district_map = dict(zip(df['district_code'], df['鄉鎮市區']))#建立地區編碼對應表
        return district_map#回傳地區對應表
    return {}#如果檔案不存在，回傳空字典

model = load_model()#載入模型
district_map = get_district_map()#載入地區對應表
name_to_code = {v: k for k, v in district_map.items()}#建立地區名稱到編碼的對應

#網頁介面設計
st.title("🔮 高雄房價「未來」預測機")#顯示網頁標題
st.markdown("### 穿越時空，查看未來房價趨勢")#顯示副標題
if model is None:#如果模型不存在
    st.error("錯誤：找不到模型檔案，請確認 house_price_model.pkl 是否存在。")#顯示錯誤訊息
    st.stop()#停止執行

# 建立輸入區塊
with st.container(border=True):#建立輸入容器
    col1, col2 = st.columns(2)#建立兩欄佈局
    with col1:#左欄
        selected_district_name = st.selectbox("📍 選擇行政區", options=sorted(name_to_code.keys()))#下拉選單選擇地區區
        district_code = name_to_code[selected_district_name]#取得選擇的行政區編碼
        house_age = st.slider("📅 目前屋齡 (年)", 0, 60, 10)#滑桿選擇屋齡
        
    with col2:#右欄
        total_ping = st.number_input("📐 權狀坪數", 5.0, 200.0, 35.0, 0.5)#輸入坪數
        total_floors = st.number_input("總樓層數", 1.0, 50.0, 15.0, 1.0)#輸入總樓層
        building_type_map = {"大樓/華廈": 3, "透天/別墅": 2, "公寓": 1}#大樓代表3，透天代表2，公寓代表1，反正就是代表編號
        selected_type = st.radio("🏢 建物型態", list(building_type_map.keys()), horizontal=True)#按鈕選擇建物型態
        building_type = building_type_map[selected_type]#取得選擇的建築物型態的號碼

    st.divider()#插入分隔線
    
    #預測年份
    target_year = st.slider("⏳ 您想預測哪一年的價格？", 2025, 2030, 2025)#滑桿選擇預測年份
    
    #計算屋齡修正 (因為到了2030年，房子屋齡也會多五年)
    future_age = house_age + (target_year - 2025)#計算未來年份的屋齡
    st.caption(f"💡 到了 {target_year} 年，這間房子的屋齡將會變成 {future_age} 年")#顯示屋齡修正資訊

#預測邏輯
if st.button("🚀 啟動時光機預測", type="primary", use_container_width=True):#按鈕觸發預測
    
    #計算使用者指定年份的價格
    input_data = pd.DataFrame([
        [
            future_age, #這裡用變老後的屋齡
            total_ping,
            total_floors,
            district_code,
            building_type,
            target_year#使用使用者選的未來年份
        ]
    ], 
    columns=['house_age', 'total_ping', 'total_floors', 'district_code', 'building_type', 'trade_year'])#建立輸入資料表
    
    pred_price_per_ping = model.predict(input_data)[0]#預測每坪價格
    total_price = pred_price_per_ping * total_ping#計算總價

    #顯示結果
    st.success(f"🗓️ 【{target_year} 年】 預測結果")#顯示預測年份
    c1, c2 = st.columns(2)#建立兩欄顯示結果
    c1.metric("預估單價", f"{pred_price_per_ping:.1f} 萬/坪")#顯示預估單價
    c2.metric("預估總價", f"{int(total_price):,} 萬元")#顯示預估總價

    #繪製未來五年走勢圖
    st.subheader("📈 未來 5 年價格趨勢模擬")#顯示子標題
    
    chart_data = []#初始化圖表資料
    years = range(2025, 2031)#定義未來年份範圍
    
    for y in years:
        # 隨著年份增加，屋齡也要跟著增加
        age_at_y = house_age + (y - 2025)#計算每年的屋齡
        
        temp_data = pd.DataFrame([
            [
                age_at_y, total_ping, total_floors, district_code, building_type, y
            ]
        ], columns=['house_age', 'total_ping', 'total_floors', 'district_code', 'building_type', 'trade_year'])#建立每年的輸入資料
        
        price = model.predict(temp_data)[0] * total_ping#預測每年的總價
        chart_data.append(int(price))#將結果加入圖表資料

    #製作圖表資料框
    chart_df = pd.DataFrame({
        "年份": [str(y) for y in years],#定義年份欄位
        "預測總價": chart_data#定義預測總價欄位
    })
    
    # 使用Streamlit內建圖表
    st.line_chart(chart_df.set_index("年份"), color="#FF4B4B")#繪製折線圖
    
    st.warning("⚠️ 注意：此預測是基於過去幾年的市場趨勢進行「線性推估」。若未來發生重大經濟變動（如政策打房、金融海嘯），實際價格可能會有落差。")#顯示警告訊息