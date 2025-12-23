"""
期貨與選擇權線上考試系統 - 優化版
主要改進：
1. JWT 身份驗證（取代不安全的 Cookie）
2. SQLite 資料庫（取代 CSV，解決並發問題）
3. RESTful API 設計（DELETE 操作改用 POST）
4. 題目快取機制
5. 考試時限功能
6. 完整日誌系統
"""

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache
import pandas as pd
import numpy as np
import sqlite3
import secrets
import logging
import os
import jwt

# ============================================================
# 設定區
# ============================================================

class Config:
    """集中管理所有設定"""
    # JWT 設定
    JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = 120  # Token 有效期 2 小時

    # 管理員認證
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

    # 考試設定
    EXAM_TIME_LIMIT_MINUTES = 60  # 考試時限（分鐘），設為 0 表示不限時
    QUESTIONS_FILE = '期中考題L.csv'
    STUDENTS_FILE = 'id.csv'
    DATABASE_FILE = 'exam_results.db'

    # Cookie 設定
    COOKIE_NAME = "exam_token"
    COOKIE_SECURE = False  # 生產環境應設為 True (HTTPS)
    COOKIE_HTTPONLY = True
    COOKIE_SAMESITE = "lax"


# ============================================================
# 日誌設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exam_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 初始化
# ============================================================

app = FastAPI(title="期貨與選擇權線上考試系統")
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# ============================================================
# 資料庫初始化（SQLite）
# ============================================================

def init_database():
    """初始化 SQLite 資料庫"""
    conn = sqlite3.connect(Config.DATABASE_FILE)
    cursor = conn.cursor()

    # 建立考試結果表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL UNIQUE,
            score REAL NOT NULL,
            correct_count INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            exam_start_time TIMESTAMP,
            exam_duration_seconds INTEGER
        )
    ''')

    # 建立考試進行中狀態表（用於追蹤開始時間）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_exams (
            student_id TEXT PRIMARY KEY,
            start_time TIMESTAMP NOT NULL,
            ip_address TEXT
        )
    ''')

    # 建立系統設定表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')

    # 初始化預設設定
    cursor.execute('''
        INSERT OR IGNORE INTO settings (key, value) VALUES ('exam_time_limit', '60')
    ''')

    conn.commit()
    conn.close()
    logger.info("資料庫初始化完成")


def get_setting(key: str, default: str = None) -> str:
    """取得系統設定"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default


def set_setting(key: str, value: str):
    """設定系統設定"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()
    logger.info(f"設定已更新: {key} = {value}")


def get_exam_time_limit() -> int:
    """取得考試時限（分鐘）"""
    value = get_setting('exam_time_limit', str(Config.EXAM_TIME_LIMIT_MINUTES))
    return int(value)


def get_exam_end_time() -> Optional[str]:
    """取得考試結束時間（ISO格式字串，如 2025-12-23T18:00）"""
    return get_setting('exam_end_time', '')


def set_exam_end_time(end_time: str):
    """設定考試結束時間"""
    set_setting('exam_end_time', end_time)


# 啟動時初始化資料庫
init_database()


# ============================================================
# JWT 工具函式
# ============================================================

def create_jwt_token(student_id: str, exam_start_time: datetime = None) -> str:
    """建立 JWT Token"""
    if exam_start_time is None:
        exam_start_time = datetime.utcnow()

    expire = datetime.utcnow() + timedelta(minutes=Config.JWT_EXPIRE_MINUTES)
    payload = {
        "sub": student_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "exam_start": exam_start_time.isoformat()
    }
    token = jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
    logger.info(f"為學號 {student_id} 建立 JWT Token")
    return token


def verify_jwt_token(token: str) -> Optional[dict]:
    """驗證 JWT Token"""
    try:
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token 已過期")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"無效的 Token: {e}")
        return None


def get_current_student(request: Request) -> Optional[dict]:
    """從請求中取得當前學生資訊"""
    token = request.cookies.get(Config.COOKIE_NAME)
    if not token:
        return None
    return verify_jwt_token(token)


# ============================================================
# 題目載入與快取
# ============================================================

@lru_cache(maxsize=1)
def load_questions_from_file() -> pd.DataFrame:
    """載入並快取題庫（只讀取一次）"""
    try:
        df = pd.read_csv(Config.QUESTIONS_FILE)
        logger.info(f"題庫載入成功，共 {len(df)} 題")
        return df
    except Exception as e:
        logger.error(f"載入題庫失敗: {e}")
        raise


@lru_cache(maxsize=1)
def load_student_ids() -> frozenset:
    """載入並快取學生名單"""
    try:
        df = pd.read_csv(Config.STUDENTS_FILE)
        student_ids = frozenset(df['id'].astype(str).values)
        logger.info(f"學生名單載入成功，共 {len(student_ids)} 人")
        return student_ids
    except Exception as e:
        logger.error(f"載入學生名單失敗: {e}")
        raise


def clear_cache():
    """清除快取（上傳新檔案時呼叫）"""
    load_questions_from_file.cache_clear()
    load_student_ids.cache_clear()
    logger.info("快取已清除")


def get_seed_from_id(student_id: str) -> int:
    """從學號取得隨機種子"""
    numbers = ''.join(filter(str.isdigit, str(student_id)))
    if numbers:
        return int(numbers)
    return sum(ord(c) for c in str(student_id))


def load_quiz_data(student_id: str):
    """根據學號載入並隨機排序題目"""
    df = load_questions_from_file().copy()
    seed = get_seed_from_id(student_id)
    np.random.seed(seed)
    random_order = np.random.permutation(len(df))
    df = df.iloc[random_order].reset_index(drop=True)

    questions = []
    for index, row in df.iterrows():
        full_text = row['題目']
        question_text = full_text.split('(A)')[0].strip()

        options = {}
        option_positions = []
        for opt in ['A', 'B', 'C', 'D']:
            pos = full_text.find(f'({opt})')
            if pos != -1:
                option_positions.append((pos, opt))

        option_positions.sort()

        for i, (pos, opt) in enumerate(option_positions):
            start = pos + 3
            if i < len(option_positions) - 1:
                end = option_positions[i + 1][0]
                options[opt] = full_text[start:end].strip()
            else:
                options[opt] = full_text[start:].strip()

        questions.append({
            '題號': index + 1,
            '原題號': random_order[index] + 1,
            '題目': question_text,
            'A': options.get('A', ''),
            'B': options.get('B', ''),
            'C': options.get('C', ''),
            'D': options.get('D', '')
        })

    questions_df = pd.DataFrame(questions)
    answers_df = pd.DataFrame({
        '題號': range(1, len(df) + 1),
        '原題號': random_order + 1,
        '正確答案': df['答案'].values
    })

    return questions_df, answers_df


# ============================================================
# 資料庫操作函式
# ============================================================

def get_db_connection():
    """取得資料庫連線"""
    conn = sqlite3.connect(Config.DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def has_completed_exam(student_id: str) -> bool:
    """檢查學生是否已完成考試"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM exam_results WHERE student_id = ?', (str(student_id),))
    result = cursor.fetchone() is not None
    conn.close()
    return result


def save_exam_result(student_id: str, score: float, correct: int, total: int,
                     ip_address: str, start_time: datetime, duration_seconds: int):
    """儲存考試結果"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO exam_results
            (student_id, score, correct_count, total_questions, ip_address, exam_start_time, exam_duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(student_id), score, correct, total, ip_address, start_time, duration_seconds))

        # 刪除進行中的考試記錄
        cursor.execute('DELETE FROM active_exams WHERE student_id = ?', (str(student_id),))

        conn.commit()
        logger.info(f"學號 {student_id} 的考試結果已儲存，分數: {score}")
    except sqlite3.IntegrityError:
        logger.warning(f"學號 {student_id} 已有考試記錄，無法重複儲存")
        raise HTTPException(status_code=403, detail="您已完成考試，不能重複作答")
    finally:
        conn.close()


def get_all_results() -> list:
    """取得所有考試結果"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT student_id, score, correct_count, total_questions,
               submission_time, ip_address, exam_duration_seconds
        FROM exam_results
        ORDER BY submission_time DESC
    ''')
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def delete_result_by_id(student_id: str):
    """刪除指定學生的考試結果"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM exam_results WHERE student_id = ?', (str(student_id),))
    conn.commit()
    conn.close()
    logger.info(f"已刪除學號 {student_id} 的考試結果")


def delete_all_results():
    """刪除所有考試結果"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM exam_results')
    cursor.execute('DELETE FROM active_exams')
    conn.commit()
    conn.close()
    logger.info("已刪除所有考試結果")


def start_exam_session(student_id: str, ip_address: str):
    """記錄考試開始"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO active_exams (student_id, start_time, ip_address)
        VALUES (?, ?, ?)
    ''', (str(student_id), datetime.now(), ip_address))
    conn.commit()
    conn.close()
    logger.info(f"學號 {student_id} 開始考試")


# ============================================================
# 學生驗證
# ============================================================

def validate_student_id(student_id: str) -> bool:
    """驗證學號是否有效"""
    try:
        valid_ids = load_student_ids()
        return str(student_id) in valid_ids
    except Exception as e:
        logger.error(f"驗證學號時發生錯誤: {e}")
        return False


# ============================================================
# 管理員驗證（使用 Cookie，不用 HTTP Basic Auth）
# ============================================================

ADMIN_COOKIE_NAME = "admin_session"
ADMIN_SESSION_TOKEN = secrets.token_urlsafe(32)  # 每次啟動產生新的 token

def verify_admin(request: Request):
    """驗證管理員身份（透過 Cookie）- 用於 Depends"""
    token = request.cookies.get(ADMIN_COOKIE_NAME)
    if not token or token != ADMIN_SESSION_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="未授權，請先登入"
        )
    return "admin"

def verify_admin_redirect(request: Request):
    """驗證管理員身份，返回 True/False"""
    token = request.cookies.get(ADMIN_COOKIE_NAME)
    if not token or token != ADMIN_SESSION_TOKEN:
        return False
    return True


# ============================================================
# 學生路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """首頁 - 登入頁面"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, student_id: str = Form(...)):
    """學生登入"""
    logger.info(f"登入嘗試: {student_id}")

    if not validate_student_id(student_id):
        logger.warning(f"無效學號登入嘗試: {student_id}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "學號無效，請重新輸入"
        })

    if has_completed_exam(student_id):
        logger.info(f"學號 {student_id} 嘗試重複考試")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "您已經完成考試，不能重複作答"
        })

    # 記錄考試開始
    client_ip = request.client.host if request.client else "unknown"
    start_exam_session(student_id, client_ip)

    # 建立 JWT Token
    exam_start_time = datetime.utcnow()
    token = create_jwt_token(student_id, exam_start_time)

    # 設定安全的 Cookie
    response = RedirectResponse(url="/quiz", status_code=303)
    response.set_cookie(
        key=Config.COOKIE_NAME,
        value=token,
        httponly=Config.COOKIE_HTTPONLY,
        secure=Config.COOKIE_SECURE,
        samesite=Config.COOKIE_SAMESITE,
        max_age=Config.JWT_EXPIRE_MINUTES * 60
    )

    logger.info(f"學號 {student_id} 登入成功")
    return response


@app.get("/quiz", response_class=HTMLResponse)
async def quiz(request: Request):
    """考試頁面"""
    student_info = get_current_student(request)

    if not student_info:
        logger.warning("未登入或 Token 過期，重導向到登入頁")
        return RedirectResponse(url="/")

    student_id = student_info["sub"]
    exam_start = datetime.fromisoformat(student_info["exam_start"])

    if has_completed_exam(student_id):
        logger.info(f"學號 {student_id} 已完成考試，重導向")
        response = RedirectResponse(url="/")
        response.delete_cookie(Config.COOKIE_NAME)
        return response

    # 取得動態時限設定
    exam_time_limit = get_exam_time_limit()
    exam_end_time_str = get_exam_end_time()

    # 計算剩餘時間（考慮時限和結束時間兩種設定）
    remaining_seconds = None
    has_time_limit = False

    # 1. 根據時限計算剩餘時間
    if exam_time_limit > 0:
        elapsed = (datetime.utcnow() - exam_start).total_seconds()
        remaining_by_limit = max(0, exam_time_limit * 60 - elapsed)
        remaining_seconds = remaining_by_limit
        has_time_limit = True

    # 2. 根據結束時間計算剩餘時間
    if exam_end_time_str:
        try:
            # 解析結束時間（本地時間）
            exam_end_time = datetime.fromisoformat(exam_end_time_str)
            # 計算距離結束時間的秒數
            remaining_by_end = (exam_end_time - datetime.now()).total_seconds()
            remaining_by_end = max(0, remaining_by_end)

            # 取兩者中較小的值
            if remaining_seconds is None:
                remaining_seconds = remaining_by_end
            else:
                remaining_seconds = min(remaining_seconds, remaining_by_end)
            has_time_limit = True
        except ValueError:
            logger.error(f"無效的結束時間格式: {exam_end_time_str}")

    # 檢查是否已超時
    if has_time_limit and remaining_seconds <= 0:
        logger.warning(f"學號 {student_id} 考試時間已到")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "考試時間已結束"
        })

    # 如果沒有任何時間限制，設為 0
    if remaining_seconds is None:
        remaining_seconds = 0

    questions_df, _ = load_quiz_data(student_id)
    questions = questions_df.to_dict('records')

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "questions": questions,
        "student_id": student_id,
        "time_limit_seconds": int(remaining_seconds),
        "has_time_limit": has_time_limit
    })


@app.post("/submit")
async def submit(request: Request):
    """提交答案"""
    student_info = get_current_student(request)

    if not student_info:
        raise HTTPException(status_code=401, detail="未登入或登入已過期")

    student_id = student_info["sub"]
    exam_start = datetime.fromisoformat(student_info["exam_start"])

    if has_completed_exam(student_id):
        raise HTTPException(status_code=403, detail="您已經完成考試，不能重複作答")

    # 檢查是否超時
    exam_time_limit = get_exam_time_limit()
    elapsed_seconds = int((datetime.utcnow() - exam_start).total_seconds())
    if exam_time_limit > 0:
        if elapsed_seconds > exam_time_limit * 60 + 60:  # 給予 60 秒緩衝
            raise HTTPException(status_code=403, detail="考試時間已結束")

    student_answers = await request.json()
    _, answers_df = load_quiz_data(student_id)

    correct_answers = answers_df.set_index('題號')['正確答案'].to_dict()
    score = 0
    total_questions = len(correct_answers)

    results = {}
    for q_num, answer in student_answers.items():
        q_num = int(q_num)
        is_correct = str(answer) == str(correct_answers.get(q_num))
        original_question_number = int(
            answers_df[answers_df['題號'] == q_num]['原題號'].iloc[0])
        results[str(q_num)] = {
            'student_answer': answer,
            'correct_answer': str(correct_answers.get(q_num)),
            'is_correct': is_correct,
            'original_question_number': original_question_number
        }
        if is_correct:
            score += 1

    final_score = (score / total_questions) * 100
    client_ip = request.client.host if request.client else "unknown"

    # 儲存到資料庫
    save_exam_result(
        student_id=student_id,
        score=final_score,
        correct=score,
        total=total_questions,
        ip_address=client_ip,
        start_time=exam_start,
        duration_seconds=elapsed_seconds
    )

    logger.info(f"學號 {student_id} 完成考試，分數: {final_score:.1f}")

    return {
        'score': float(final_score),
        'correct': int(score),
        'total': int(total_questions),
        'duration_seconds': elapsed_seconds,
        'results': results
    }


@app.get("/logout")
async def logout():
    """學生登出"""
    response = RedirectResponse(url="/")
    response.delete_cookie(Config.COOKIE_NAME)
    return response


# ============================================================
# 管理員路由
# ============================================================

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, error: str = None):
    """管理員登入頁面"""
    # 如果已登入，重導向到後台
    if verify_admin_redirect(request):
        return RedirectResponse(url="/admin", status_code=303)

    login_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>管理員登入</title>
        <style>
            body { font-family: "Microsoft JhengHei", Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .login-box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); width: 300px; }
            h2 { margin: 0 0 20px 0; text-align: center; color: #333; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; color: #666; }
            input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background: #5a6fd6; }
            .error { color: #f44336; text-align: center; margin-bottom: 15px; }
            .back-link { text-align: center; margin-top: 15px; }
            .back-link a { color: #667eea; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 管理員登入</h2>
            ''' + (f'<div class="error">{error}</div>' if error else '') + '''
            <form method="POST" action="/admin/login">
                <div class="form-group">
                    <label>帳號</label>
                    <input type="text" name="username" required autofocus>
                </div>
                <div class="form-group">
                    <label>密碼</label>
                    <input type="password" name="password" required>
                </div>
                <button type="submit">登入</button>
            </form>
            <div class="back-link"><a href="/">← 返回首頁</a></div>
        </div>
    </body>
    </html>
    '''
    return HTMLResponse(content=login_html)

@app.post("/admin/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """處理管理員登入"""
    if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(
            key=ADMIN_COOKIE_NAME,
            value=ADMIN_SESSION_TOKEN,
            httponly=True,
            max_age=3600 * 8  # 8 小時
        )
        logger.info("管理員登入成功")
        return response
    else:
        logger.warning(f"管理員登入失敗，嘗試的帳號: {username}")
        return RedirectResponse(url="/admin/login?error=帳號或密碼錯誤", status_code=303)

@app.get("/admin")
async def admin_dashboard(request: Request):
    """管理員儀表板"""
    # 檢查是否已登入
    if not verify_admin_redirect(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    results = get_all_results()

    # 計算統計資料
    if results:
        stats = {
            'total_students': len(results),
            'average_score': sum(r['score'] for r in results) / len(results),
            'pass_rate': sum(1 for r in results if r['score'] >= 60) / len(results) * 100
        }
    else:
        stats = {'total_students': 0, 'average_score': 0, 'pass_rate': 0}

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "results": results,
        "stats": stats,
        "admin_logout_url": "/admin/logout",
        "exam_time_limit": get_exam_time_limit(),
        "exam_end_time": get_exam_end_time()
    })


@app.post("/admin/settings/time-limit")
async def update_time_limit(time_limit: int = Form(...), _: str = Depends(verify_admin)):
    """更新考試時限"""
    if time_limit < 0:
        raise HTTPException(status_code=400, detail="時限不能為負數")
    set_setting('exam_time_limit', str(time_limit))
    logger.info(f"考試時限已更新為 {time_limit} 分鐘")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/settings/end-time")
async def update_end_time(end_time: str = Form(""), _: str = Depends(verify_admin)):
    """更新考試結束時間"""
    set_exam_end_time(end_time)
    if end_time:
        logger.info(f"考試結束時間已設定為 {end_time}")
    else:
        logger.info("考試結束時間已清除")
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/download/results")
async def download_results(_: str = Depends(verify_admin)):
    """下載考試結果"""
    results = get_all_results()
    if not results:
        raise HTTPException(status_code=404, detail="尚無考試結果")

    df = pd.DataFrame(results)
    csv_path = 'temp_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return FileResponse(csv_path, filename='考試結果.csv', media_type='text/csv')


@app.get("/admin/download/questions")
async def download_questions(_: str = Depends(verify_admin)):
    """下載題庫"""
    return FileResponse(Config.QUESTIONS_FILE, filename='期中考題.csv')


@app.get("/admin/download/students")
async def download_students(_: str = Depends(verify_admin)):
    """下載學生名單"""
    return FileResponse(Config.STUDENTS_FILE, filename='學生名單.csv')


@app.get("/admin/download/template/questions")
async def download_questions_template(_: str = Depends(verify_admin)):
    """下載題庫空白範本"""
    from fastapi.responses import Response
    from urllib.parse import quote
    # 範本包含範例，格式與實際題庫一致
    template_content = '題目,答案\n"範例題目：下列何者正確？ (A)選項A\t(B)選項B\t(C)選項C\t(D)選項D",A\n範例題目二：請選擇正確答案(A)答案一(B)答案二(C)答案三(D)答案四,B\n'
    filename = "題庫範本.csv"
    encoded_filename = quote(filename)
    return Response(
        content=template_content.encode('utf-8-sig'),
        media_type='text/csv',
        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{encoded_filename}"}
    )


@app.get("/admin/download/template/students")
async def download_students_template(_: str = Depends(verify_admin)):
    """下載學生名單空白範本"""
    from fastapi.responses import Response
    from urllib.parse import quote
    # 範本包含範例，格式與實際名單一致
    template_content = 'id,password\nA12345678,A12345678\nB23456789,B23456789\n'
    filename = "學生名單範本.csv"
    encoded_filename = quote(filename)
    return Response(
        content=template_content.encode('utf-8-sig'),
        media_type='text/csv',
        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{encoded_filename}"}
    )


# 改用 POST 方法處理刪除操作（安全性優化）
@app.post("/admin/delete/result/{student_id}")
async def delete_result_api(student_id: str, _: str = Depends(verify_admin)):
    """刪除指定學生的成績"""
    delete_result_by_id(student_id)
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/delete/all-results")
async def delete_all_results_api(_: str = Depends(verify_admin)):
    """刪除所有成績"""
    delete_all_results()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/upload/questions")
async def upload_questions(questions_file: UploadFile = File(...), _: str = Depends(verify_admin)):
    """上傳新題庫"""
    if not questions_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="只接受 CSV 檔案")

    content = await questions_file.read()
    with open(Config.QUESTIONS_FILE, 'wb') as f:
        f.write(content)

    clear_cache()  # 清除快取以載入新題庫
    logger.info(f"題庫已更新: {questions_file.filename}")

    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/upload/students")
async def upload_students(students_file: UploadFile = File(...), _: str = Depends(verify_admin)):
    """上傳新學生名單"""
    if not students_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="只接受 CSV 檔案")

    content = await students_file.read()
    with open(Config.STUDENTS_FILE, 'wb') as f:
        f.write(content)

    clear_cache()  # 清除快取以載入新名單
    logger.info(f"學生名單已更新: {students_file.filename}")

    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/logout")
async def admin_logout():
    """管理員登出"""
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    logger.info("管理員已登出")
    return response


# ============================================================
# 相容性路由（保持舊 GET 刪除路由可用）
# ============================================================

@app.get("/admin/delete/result/{student_id}")
async def delete_result_get(student_id: str, _: str = Depends(verify_admin)):
    """GET 刪除路由（已棄用，但保持相容）"""
    logger.warning(f"使用已棄用的 GET 刪除路由: /admin/delete/result/{student_id}")
    delete_result_by_id(student_id)
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/delete/all-results")
async def delete_all_results_get(_: str = Depends(verify_admin)):
    """GET 刪除全部路由（已棄用，但保持相容）"""
    logger.warning("使用已棄用的 GET 刪除路由: /admin/delete/all-results")
    delete_all_results()
    return RedirectResponse(url="/admin", status_code=303)


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("啟動考試系統...")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
