from data_processor import prepare_and_split  # 從自訂的資料處理模組匯入準備與切割資料的函式
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # 匯入隨機森林與梯度提升迴歸模型
from sklearn.metrics import mean_squared_error, r2_score  # 匯入評估模型表現的指標
import joblib  # 匯入儲存與載入模型的工具
import numpy as np  # 匯入數值運算的工具

def train_model():
    print("🚀 啟動高階模型訓練 (X-Mode)...")  # 印出訓練開始的訊息
    
    X_train, X_test, y_train, y_test = prepare_and_split()  # 準備訓練資料與測試資料
    
    if X_train is None:  # 如果訓練資料為空
        print("❌ 無法開始訓練")  # 印出錯誤訊息
        return  # 結束函式

    print(f"訓練資料筆數: {len(X_train)}")  # 印出訓練資料的筆數
    print("🔥 正在使用 Gradient Boosting 進行深度學習 (會比之前久一點點)...")  # 印出目前使用的模型為梯度提升模型

    # 🌟 升級版演算法：Gradient Boosting
    #這比原本的 Random Forest 更強，更能抓到數據的細節
    model = GradientBoostingRegressor(
        n_estimators=1000,     # 提高決策樹數量至 1000 棵以提升精度
        learning_rate=0.05,    # 設定較低的學習率以進行細緻化學習
        max_depth=6,           # 限制樹深，防止過度擬合 (Overfitting)
        min_samples_leaf=5,    # 防止過度擬合
        random_state=42
    )
    
    model.fit(X_train, y_train)  # 用訓練資料訓練模型
    
    # 預測與評估
    predictions = model.predict(X_test)  # 用訓練好的模型對測試資料進行預測
    rmse = np.sqrt(mean_squared_error(y_test, predictions))  # 計算均方根誤差
    r2 = r2_score(y_test, predictions)  # 計算決定係數
    
    print("\n📊 --- 最終成績單 ---")
    print(f"RMSE (誤差): {rmse:.2f} 萬元/坪")  # 印出誤差
    print(f"R2 Score : {r2:.4f}")  # 印出決定係數
    
    if r2 > 0.75:  # 如果決定係數大於 0.75
        print("🏆 太強了！這已經是專業分析師等級的準確度！")  # 印出優異表現的訊息
    elif r2 > 0.65:  # 如果決定係數介於 0.65 與 0.75 之間
        print("👍 很棒！模型已經能抓到大部分的房價規律。")  # 印出良好表現的訊息
    else:  # 其他情況
        print("💪 加油，資料可能太雜亂，我們再試試看。")  # 印出需要加油的訊息

    # 儲存模型
    joblib.dump(model, 'house_price_model.pkl')  # 將訓練好的模型儲存至檔案
    print("\n💾 超級模型已儲存。")  # 印出儲存完成的訊息

if __name__ == "__main__":
    train_model()  # 如果是主程式，執行訓練模型的函式