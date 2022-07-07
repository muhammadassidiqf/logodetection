import pymysql.cursors, os
# from logo_db.logo_table import Hasil
# from flask import flash, render_template, request, redirect
# from werkzeug.security import generate_password_hash, check_password_hash

class LogoDB():
    def __init__(self, logo_nama, logo_filename, logo_path):
        self.logo_path = logo_path
        self.logo_nama = logo_nama
        self.logo_filename = logo_filename
   
    conn = cursor = None
    def open_db(self):
        global conn, cursor
        conn = pymysql.connect(host='localhost',user='root',password='',database='logodetection',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
        cursor = conn.cursor()	
    def close_db(self):
        global conn, cursor
        cursor.close()
        conn.close()
    def add_logo(self, logo_nama, logo_filename, logo_path):
        try: 
            self.open_db(self)
            sql = "INSERT INTO logo(logo_nama, logo_filename, logo_path) VALUES (%s, %s, %s)"
            val = (logo_nama, logo_filename, logo_path)
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
    def login_check(self, username, password):
        try: 
            self.open_db(self)
            sql = "SELECT * FROM users WHERE user_name = %s AND user_password=%s"
            val = (username, password)
            cursor.execute(sql, val)
            results = cursor.fetchone()
            self.close_db(self)
            return results
        except Exception as e:
            return e
    def show_logo(self):
        data_show = []
        try:
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num, logo.* FROM logo"
            cursor.execute(sql)
            results = cursor.fetchall()
            
            for data in results:
                data_show.append(data)
            
            self.close_db(self)
            return data_show
        except Exception as e:
            return e

    def get_one_logo(self, logo_id):
        try:
            self.open_db(self)
            sql = "SELECT * FROM logo WHERE logo_id = %s"
            val = (logo_id)
            cursor.execute(sql, val)
            results = cursor.fetchone()
            self.close_db(self)
            return results
        except Exception as e:
            return e

    def get_all_model(self):
        data_show = []
        try: 
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num,model.* FROM model"
            cursor.execute(sql)
            results = cursor.fetchall()
            for data in results:
                data_show.append(data)
            self.close_db(self)
            return data_show
        except Exception as e:
            return e

    def get_all_video(self):
        data_show = []
        try: 
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num, model.model_nama, video.*, video.created_at as uploaded_at, (Select sum(outputs.ads_per_menit) as total_ads from outputs where outputs.video_id = video.video_id) as total_ads FROM video JOIN model ON model.model_id = video.model_id JOIN users ON users.user_id = video.user_id"
            cursor.execute(sql)
            results = cursor.fetchall()
            for data in results:
                data_show.append(data)
            self.close_db(self)
            return data_show
        except Exception as e:
            return e

    def get_one_model(self, model_id):
        try: 
            self.open_db(self)
            sql = "SELECT * FROM model WHERE model_id = %s"
            val = (model_id)
            cursor.execute(sql, val)
            results = cursor.fetchone()
            self.close_db(self)
            return results
        except Exception as e:
            return e
    def get_one_video(self, video_id):
        try: 
            self.open_db(self)
            sql = "SELECT * FROM video WHERE video_filename_akhir = %s"
            val = (video_id)
            cursor.execute(sql, val)
            results = cursor.fetchone()
            self.close_db(self)
            return results
        except Exception as e:
            return e

    def add_durasi(self, arr_durasi):
        try: 
            self.open_db(self)
            sql = "INSERT INTO outputs(video_id, start_time, end_time, durasi, durasi_sec, ads_per_menit) VALUES (%s, %s, %s, %s, %s, %s)"
            val = arr_durasi
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
    def add_video(self, arr_video):
        try: 
            self.open_db(self)
            sql = "INSERT INTO video(video_filename_awal, video_filename_akhir, model_id, ads_rate,user_id) VALUES (%s, %s, %s, %s,%s)"
            val = arr_video
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
    def add_model(self, arr_model):
        try: 
            self.open_db(self)
            sql = "INSERT INTO model(model_nama, model_weights, model_cfg, model_label) VALUES (%s, %s, %s, %s)"
            val = arr_model
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
    def delete_model(self, id_model):
        try: 
            self.open_db(self)
            sql = "Delete from model where model_id = %s"
            val = id_model
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
    def get_output(self, video_id):
        data_show = []
        try: 
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num, outputs.*, model.model_nama FROM outputs JOIN video ON video.video_id=outputs.video_id JOIN model ON model.model_id = video.model_id WHERE video.video_filename_akhir = %s"
            val = (video_id)
            cursor.execute(sql, val)
            results = cursor.fetchall()
            for data in results:
                data_show.append(data)
            self.close_db(self)
            return data_show
        except Exception as e:
            return e
    def get_output_by_videoid(self, video_id):
        data_show = []
        try: 
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num, outputs.*, model.model_nama FROM outputs JOIN video ON video.video_id=outputs.video_id JOIN model ON model.model_id = video.model_id WHERE video.video_id = %s"
            val = (video_id)
            cursor.execute(sql, val)
            results = cursor.fetchall()
            for data in results:
                data_show.append(data)
            self.close_db(self)
            return data_show
        except Exception as e:
            return e
    def get_video_by_userid(self, user_id):
        data_show = []
        try: 
            self.open_db(self)
            sql = "SELECT ROW_NUMBER() OVER() AS num, model.model_nama, video.*, video.created_at as uploaded_at, (Select sum(outputs.ads_per_menit) as total_ads from outputs where outputs.video_id = video.video_id) as total_ads FROM video JOIN model ON model.model_id = video.model_id JOIN users ON users.user_id = video.user_id WHERE video.user_id = %s"
            val = (user_id)
            cursor.execute(sql, val)
            results = cursor.fetchall()
            for data in results:
                data_show.append(data)
            self.close_db(self)
            return data_show
        except Exception as e:
            return e
    def get_video_by_videoid(self, video_id):
        try: 
            self.open_db(self)
            sql = "SELECT * FROM video WHERE video_id = %s"
            val = (video_id)
            cursor.execute(sql, val)
            results = cursor.fetchone()
            self.close_db(self)
            return results
        except Exception as e:
            return e
    def add_estimated_time(self, data):
        try: 
            self.open_db(self)
            sql = "UPDATE video SET video_estimasi = %s WHERE video_filename_akhir = %s"
            val = data
            cursor.execute(sql, val)
            conn.commit()
            self.close_db(self)
            return 'berhasil'
        except Exception as e:
            return e
         
if __name__ == "__main__":
    logo_nama = 'test'
    logo_filename = 'test'
    logo_path = 'test'
    # hasil_add = LogoDB.add_logo(self=LogoDB, logo_nama=logo_nama, logo_filename=logo_filename, logo_path=logo_path)
    getmodel = LogoDB.add_durasi(self=LogoDB, arr_durasi=[1,'1:05','1:10','0:5','5'])
    print(getmodel)
    # print(getmodel['model_nama'])