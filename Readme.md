## **📌 Project Title**

```
ChurnGuard AI – Customer Retention Capital Allocation Engine
```

---

## **🧭 Real Business Problem**

Subscription-based companies often lose 15–30% of customers annually.

However:

> Retaining every customer is financially inefficient

Retention campaigns involve:

* Discounts
* Loyalty credits
* Account manager effort
* Marketing outreach

Which incurs **retention cost**.

The real business question is NOT:

> Who will churn?

But:

> Which customer segments should we invest retention capital in?

This project converts churn predictions into:

✔ Revenue at Risk

✔ Retention Investment Required

✔ Recoverable Revenue

✔ Net Value Created

✔ CFO-grade Budget Allocation Strategy

---

## **🎯 Purpose of This Project**

To build a decision-intelligence system that enables:

Finance teams to determine:

```
Where should retention budget be allocated
to maximise recoverable revenue?
```

Instead of:

```
Blindly running retention campaigns
across all high-risk customers
```

---

## **⚙️ Key Capabilities**

* Customer churn prediction
* Risk-tier segmentation
* Revenue exposure estimation
* Retention investment simulation
* Recoverable revenue modelling
* ROI-driven intervention strategy
* CFO-ready PDF report
* Boardroom-ready PPT deck

---

## **🧱 Tech Stack**

| **Layer**  | **Tool** |
| ---------------- | -------------- |
| Inference Engine | Python         |
| ML Model         | XGBoost        |
| Dashboard        | Streamlit      |
| Data Processing  | Pandas         |
| Reporting        | ReportLab      |
| Presentation     | python-pptx    |

---

## **🚀 How to Run This Project**

---

### **Step 1 – Clone Repository**

```
git clone https://github.com/<your-username>/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

---

### **Step 2 – Create Virtual Environment**

```
python -m venv .venv
```

Activate:

Mac/Linux:

```
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

---

### **Step 3 – Install Dependencies**

```
pip install -r requirements.txt
```

---

### **Step 4 – Run Application**

```
streamlit run app.py
```

---

### **Step 5 – Upload CSV**

Upload customer dataset containing:

* tenure
* MonthlyCharges
* Contract
* gender
* etc.

The system will:

✔ Predict churn

✔ Segment customers

✔ Simulate retention campaign

✔ Generate CFO report

✔ Generate Boardroom PPT

---

## **🧪 How to Modify for Custom Dataset**

1. Update training schema:

```
models/training_schema.json
```

2. Ensure uploaded CSV follows same feature structure
3. Update preprocessing rules:

```
src/data/preprocess.py
```

## Dataset (Used for Training Model)

Due to size constraints, the dataset is not included in the repository.

Download it from:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place it in:

data/raw/

---

## **⚠️ Challenges Faced During Development**

| **Challenge**                 | **Resolution**                 |
| ----------------------------------- | ------------------------------------ |
| Inference schema mismatch           | Implemented dynamic schema alignment |
| Categorical value inconsistency     | Category normalization layer         |
| Numeric dtype failures              | Training-dtype enforcement           |
| Missing engineered features         | Runtime recreation (tenure_group)    |
| Prediction failure due to OHE drift | Pipeline-schema alignment            |
| Finance reporting mismatch          | ROI-driven segment modelling         |

---

## **📚 Key Learnings**

* Prediction ≠ Business Decision
* Model accuracy does not guarantee ROI
* Retention campaigns must be budget-constrained
* Financial impact modelling is essential for adoption
* Schema alignment is critical in production ML systems

---

## **🏢 Intended Enterprise Use**

This system can assist:

* Telecom companies
* SaaS subscription platforms
* Insurance providers
* OTT services
* Fintech platforms

in:

Retention strategy planning

Budget optimisation

Audit-ready campaign justification

---

## **📊 Output Deliverables**

* CFO Financial Impact Report (PDF)
* Boardroom Strategy Deck (PPTX)

---
