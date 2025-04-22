from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import os

# 读取模型
model = joblib.load("model/churn_model.pkl")

# API 初始化
app = FastAPI(title="电信客户流失预测API")
FEATURE_COLUMNS = [
    "region", "dual_band", "refurbished", "current_phone_price", "phone_network_function",
    "marital_status", "num_adults", "info_match", "expected_income", "credit_card",
    "device_usage_days", "total_work_months", "only_subscriber_in_family", "active_users_in_family",
    "new_phone_user", "credit_score", "avg_monthly_fee", "avg_monthly_usage_mins",
    "avg_overuse_mins", "avg_overuse_fee", "avg_voice_fee", "avg_data_overload_fee",
    "avg_roaming_calls", "monthly_usage_pct_change", "monthly_fee_pct_change",
    "avg_dropped_voice_calls", "avg_dropped_data_calls", "avg_busy_voice_calls",
    "avg_busy_data_calls", "avg_missed_voice_calls", "avg_unanswered_data_calls",
    "avg_outgoing_voice_calls", "avg_outgoing_data_calls", "avg_incoming_voice_calls",
    "avg_completed_voice_calls", "avg_completed_data_calls", "avg_cust_service_calls",
    "avg_cust_service_call_minutes", "avg_1min_incoming_calls", "avg_three_way_calls",
    "avg_completed_voice_call_minutes", "avg_peak_voice_calls", "avg_peak_data_calls",
    "avg_peak_incomplete_voice_mins", "avg_offpeak_voice_calls", "avg_offpeak_data_calls",
    "avg_dropped_or_busy_calls", "avg_attempted_calls", "avg_completed_calls",
    "avg_call_forwarding_calls", "avg_call_waiting_calls", "account_spending_limit",
    "total_calls_lifetime", "total_minutes_lifetime", "total_fee_lifetime", "adjusted_total_fee",
    "adjusted_total_minutes", "adjusted_total_calls", "lifetime_avg_monthly_fee",
    "lifetime_avg_monthly_mins", "lifetime_avg_monthly_calls", "avg_mins_3months",
    "avg_calls_3months", "avg_fee_3months", "avg_mins_6months", "avg_calls_6months", "avg_fee_6months"
]


# 输入字段模型
class CustomerInput(BaseModel):
    customer_id: int
    region: int
    dual_band: int
    refurbished: int
    current_phone_price: int
    phone_network_function: int
    marital_status: int
    num_adults: int
    info_match: int
    expected_income: int
    credit_card: int
    device_usage_days: int
    total_work_months: int
    only_subscriber_in_family: int
    active_users_in_family: int
    new_phone_user: int
    credit_score: int
    avg_monthly_fee: float
    avg_monthly_usage_mins: float
    avg_overuse_mins: float
    avg_overuse_fee: float
    avg_voice_fee: float
    avg_data_overload_fee: float
    avg_roaming_calls: float
    monthly_usage_pct_change: float
    monthly_fee_pct_change: float
    avg_dropped_voice_calls: float
    avg_dropped_data_calls: float
    avg_busy_voice_calls: float
    avg_busy_data_calls: float
    avg_missed_voice_calls: float
    avg_unanswered_data_calls: float
    avg_outgoing_voice_calls: float
    avg_outgoing_data_calls: float
    avg_incoming_voice_calls: float
    avg_completed_voice_calls: float
    avg_completed_data_calls: float
    avg_cust_service_calls: float
    avg_cust_service_call_minutes: float
    avg_1min_incoming_calls: float
    avg_three_way_calls: float
    avg_completed_voice_call_minutes: float
    avg_peak_voice_calls: float
    avg_peak_data_calls: float
    avg_peak_incomplete_voice_mins: float
    avg_offpeak_voice_calls: float
    avg_offpeak_data_calls: float
    avg_dropped_or_busy_calls: float
    avg_attempted_calls: float
    avg_completed_calls: float
    avg_call_forwarding_calls: float
    avg_call_waiting_calls: float
    account_spending_limit: float
    total_calls_lifetime: float
    total_minutes_lifetime: float
    total_fee_lifetime: float
    adjusted_total_fee: float
    adjusted_total_minutes: float
    adjusted_total_calls: float
    lifetime_avg_monthly_fee: float
    lifetime_avg_monthly_mins: float
    lifetime_avg_monthly_calls: float
    avg_mins_3months: float
    avg_calls_3months: float
    avg_fee_3months: float
    avg_mins_6months: float
    avg_calls_6months: float
    avg_fee_6months: float


@app.post("/predict")
def predict_churn(data: CustomerInput):
    try:
        # 将输入数据转换为特征矩阵
        X = np.array([[getattr(data, field) for field in FEATURE_COLUMNS]])
        print("🧪 输入维度:", X.shape)

        # 使用模型进行预测
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][int(pred)]

        # 根据概率得出风险评分
        risk = "高" if prob > 0.7 else ("中" if prob > 0.4 else "低")
        return {
            "customer_id": data.customer_id,
            "预测流失": "是" if pred == 1 else "否",
            "流失概率": round(prob, 3),
            "风险评分": risk
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


# CSV上传预测接口
@app.post("/upload/csv_predict")
async def upload_csv_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # 确保CSV文件包含所有必要的特征列
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_columns:
            return {"error": f"CSV文件缺少以下列: {', '.join(missing_columns)}"}

        # 进行预测
        X = df[FEATURE_COLUMNS].values
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = []
        for i, pred in enumerate(predictions):
            prob = probabilities[i][int(pred)]
            risk = "高" if prob > 0.7 else ("中" if prob > 0.4 else "低")
            results.append({
                "customer_id": df.iloc[i]["customer_id"],
                "预测流失": "是" if pred == 1 else "否",
                "流失概率": round(prob, 3),
                "风险评分": risk
            })

        return {"predictions": results}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


# 模型重训接口
@app.post("/retrain_model")
async def retrain_model(file: UploadFile = File(...)):
    try:
        # 读取上传的CSV文件并重新训练模型
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # 需要确保数据有特征列和标签列（假设标签列名为'CHURN'）
        if "CHURN" not in df.columns:
            return {"error": "CSV文件中缺少 'CHURN' 列作为标签"}

        # 拆分数据集为训练和测试
        X = df[FEATURE_COLUMNS]
        y = df["CHURN"]

        # 拆分为训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)

        # 保存新模型
        joblib.dump(new_model, "app/models/churn_model.pkl")

        return {"message": "模型已重新训练并保存成功"}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
