# 期貨與選擇權線上考試系統

基於 FastAPI 的線上考試系統，支援學生線上作答、自動計分、管理員後台管理。

## 功能特色

- **學生端**：學號登入、線上作答、即時顯示成績
- **管理員端**：查看成績、下載報表、刪除紀錄、設定考試時限
- **安全性**：JWT 身份驗證、SQLite 資料庫（解決並發問題）
- **彈性設定**：環境變數管理、可配置考試時限

## 快速開始

### 1. 安裝依賴

```bash
# 建立虛擬環境（推薦）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
# 複製範本
cp .env.example .env

# 編輯 .env 檔案
```

`.env` 內容：
```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_password
JWT_SECRET=your_secret_key
```

### 3. 準備資料檔案

#### 學生名單 `id.csv`
```csv
id,password
F1140001,F1140001
F1140002,F1140002
```

#### 題庫 `期中考題L.csv`
```csv
題目,答案
"題目內容... (A)選項A (B)選項B (C)選項C (D)選項D",A
"題目內容...",B
```

### 4. 啟動伺服器

```bash
python main.py
```

開啟瀏覽器訪問：
- 學生登入：http://localhost:8000
- 管理員後台：http://localhost:8000/admin

---

## 部署到 Zeabur

### 1. 連結 GitHub Repo

在 Zeabur Dashboard 中選擇 **Deploy from GitHub**，選取此專案。

### 2. 設定環境變數

在 **Variables** 頁面加入：

| Key | Value | 說明 |
|-----|-------|------|
| `PRODUCTION` | `true` | 啟用生產模式 |
| `ADMIN_USERNAME` | `your_admin` | 管理員帳號 |
| `ADMIN_PASSWORD` | `StrongP@ss123` | 管理員密碼（請用強密碼） |
| `JWT_SECRET` | `64位元隨機字串` | JWT 簽章密鑰 |
| `PORT` | `8000` | 服務埠號（Zeabur 通常自動設定） |

> 產生 JWT_SECRET：`openssl rand -hex 32`

### 3. 上傳資料檔案

部署後需上傳：
- `id.csv` - 學生名單
- `期中考題L.csv` - 題庫檔案

---

## 專案結構

```
.
├── main.py              # 主程式
├── requirements.txt     # Python 依賴
├── .env.example         # 環境變數範本
├── .env                 # 本地環境變數（不納入版控）
├── id.csv               # 學生名單
├── 期中考題L.csv         # 題庫
├── exam_results.db      # SQLite 資料庫（自動產生）
├── exam_system.log      # 系統日誌
└── templates/           # HTML 模板
    ├── login.html
    ├── quiz.html
    ├── admin_login.html
    └── admin.html
```

---

## 環境變數說明

| 變數名 | 預設值 | 說明 |
|--------|--------|------|
| `ADMIN_USERNAME` | `admin` | 管理員帳號 |
| `ADMIN_PASSWORD` | `admin123` | 管理員密碼 |
| `JWT_SECRET` | 隨機產生 | JWT 簽章密鑰 |
| `PORT` | `8000` | 服務埠號 |
| `PRODUCTION` | - | 設為 `true` 啟用生產模式檢查 |

---

## 管理員功能

- **查看成績**：即時查看所有學生作答結果
- **下載報表**：匯出 CSV 格式成績單
- **刪除紀錄**：允許學生重新作答
- **時限設定**：設定考試時間限制（0 = 不限時）

---

## 開發說明

```bash
# 啟動開發模式（自動重載）
python main.py

# 查看日誌
tail -f exam_system.log
```

## License

MIT
