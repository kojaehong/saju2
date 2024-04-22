import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from flask_talisman import Talisman
import pymysql
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)  # 모든 도메인의 접근을 허용
Talisman(app)  # 기본 보안 설정 활성화

# 모델 초기화
model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 한국어 모델 사용 유지

# 데이터베이스 연결 설정
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        db=os.getenv('DB_NAME'),
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route('/')
def index():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM temp_que_ans where wr_id=1")
            results = cursor.fetchall()
            return jsonify(results)
    finally:
        conn.close()

@app.route('/osan_csv_kor_emd/', methods=['GET', 'POST'])
def osan_csv_kor_emd():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT wr_id, que FROM temp_que_ans WHERE CHAR_LENGTH(emb) < 10 OR emb IS NULL;")
            rows = cursor.fetchall()

            if not rows:
                return jsonify({"message": "No entries need updating."}), 200

            updated_rows = 0
            for row in rows:
                wr_id = row['wr_id']
                que = row['que']
                embeddings = model.encode([que])
                embedding_str = json.dumps(embeddings[0].tolist())
                cursor.execute("UPDATE temp_que_ans SET emb = %s WHERE wr_id = %s", (embedding_str, wr_id))
                updated_rows += 1

            conn.commit()
            return jsonify({"message": f"Embeddings updated successfully for {updated_rows} entries."}), 200
    finally:
        conn.close()

@app.route('/saju2/', methods=['POST'])
def saju2():
    text = request.form.get('title', '')
    f_obj = request.form.get('f_obj', '')
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql_query = f"SELECT wr_id, emb, key_01, que, ans, key_02 FROM temp_que_ans WHERE key_01 = '{f_obj}'"
            cursor.execute(sql_query)
            entries = cursor.fetchall()

            if not entries:
                return jsonify({"error": "No matching records found"}), 404

            user_embedding = model.encode(text)
            max_similarity = -1
            best_entry = None
            for entry in entries:
                # print(f"Debug: Entry fetched {entry}")  # 로깅 추가
                db_embedding = json.loads(entry['emb'])
                similarity = cosine_similarity([user_embedding], [db_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_entry = entry

            if not best_entry:
                return jsonify({"error": "No valid responses found"}), 404

            # 결과 로깅
            # print(f"Debug: Best entry {best_entry}")  # 로깅 추가

            response = {
                'a': best_entry['key_01'],
                'b': best_entry['que'],
                'c': best_entry['ans'],
                'wr_id': best_entry['wr_id'],
                'key_02': int(best_entry['key_02']),  # 데이터 타입 변환 확인
                'distance': max_similarity
            }
        
            return jsonify(response)
    except Exception as e:
        # print(f"Error: {str(e)}")  # 오류 로깅
        return jsonify({"error": "Server error", "details": str(e)}), 500
    finally:
        conn.close()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
