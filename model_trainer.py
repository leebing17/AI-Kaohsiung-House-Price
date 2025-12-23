from data_processor import prepare_and_split  #從自訂的資料處理模組匯入準備與切割資料的函式
from sklearn.ensemble import GradientBoostingRegressor  #匯入與梯度提升迴歸模型
from sklearn.metrics import mean_squared_error, r2_score  #匯入評估模型表現的指標
import joblib  #匯入儲存與載入模型的工具
import numpy as np#匯入數值運算的工具

def train_model():
    print("啟動模型訓練 (X-Mode)...")  #印出訓練開始的訊息
    
    X_train, X_test, y_train, y_test = prepare_and_split()  #準備訓練資料與測資
    
    if X_train is None:  #如果訓練資料為空
        print("無法開始訓練")  #印出錯誤訊息
        return  #結束函式

    print(f"訓練資料筆數: {len(X_train)}")  #印出訓練資料的筆數
    print("正在使用梯度提升迴歸進行深度學習")  #印出目前使用的模型為梯度提升模型
    #梯度提升回歸演算法
    model = GradientBoostingRegressor(
        n_estimators=1000,  #提高決策樹數量至 1000棵以提升精度
        learning_rate=0.01, #設定較低的學習率以進行比較細一點的學習
        max_depth=6,        #限制樹深，防止過度擬合
        min_samples_leaf=5, #防止過度擬合
        random_state=42
    )
    
    model.fit(X_train, y_train) #用訓練資料訓練模型
    
    #預測與評估
    predictions = model.predict(X_test)  #用訓練好的模型對測資進行預測
    rmse = np.sqrt(mean_squared_error(y_test, predictions))#計算均方根誤差
    r2 = r2_score(y_test, predictions)  #計算決定係數
    
    print("\n最終結果")#印出最終結果
    print(f"RMSE (誤差): {rmse:.2f} 萬元/坪")#印出誤差
    print(f"R2 Score : {r2:.4f}")#印出決定係數

    # 儲存模型
    joblib.dump(model,'house_price_model.pkl')#將訓練好的模型儲存至檔案
    print("\n模型已儲存。")  # 印出儲存完成的訊息

if __name__ == "__main__":
    train_model()#如果是主程式，執行訓練模型的函式