from distutils.log import Log
from posixpath import split
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response, send_from_directory
from werkzeug.utils import secure_filename
from inspect import ismethod
# from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy.sql import text
import datetime
import time
import os
import random
from logo_db.main import LogoDB
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import io
import xlwt


# from Video import Video
from logo_db.YoloDetector import YoloDetector

app = Flask(__name__)
app.secret_key = '937b33f1db0be340898fbac0cd8723ad337549c67f769f3040f30c7a0eab0c00'
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["RESULT_FOLDER"] = "results/"
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg", "JPG", "mp4"])

def gen_frames(video):
    cap = cv2.VideoCapture(video)
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.png', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


def run(obj, *args, **kwargs):
    for name in dir(obj):
        attribute = getattr(obj, name)
        if ismethod(attribute):
            attribute(*args, **kwargs)

# def prepare_image(file):
#     img = image.load_img(file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_norm = image.img_to_array(img).astype(np.float32)/255
#     img_array_expanded_dims = np.expand_dims(img_norm, axis=0)
#     return img_array_expanded_dims

# def prediction(filepath):
#     model = load_model('static/models/bca1.hdf5')
#     model.compile(loss='categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])

#     #matriks citra asli
#     inp = image.load_img(filepath)
#     img_array = image.img_to_array(inp)

#     start = time.time()

#     #matriks pre-processing
#     img = prepare_image(filepath)

#     #proses prediksi
#     classes = model.predict(img)

#     timing = time.time() - start

#     print("processing time: ", timing)


#     res = []   
#     if classes[0][0] == 1:
#         result = classes[0][0] * 100
#         res.append((result,img_array,img))
#         # print("Daun Padi Sehat")
#     elif classes[0][1] > classes[0][0]:
#         result = classes[0][1] * 100
#         res.append((result,img_array,img))
#         # print("Penyakit Blas Daun Padi")
#     return res

@app.route("/index", methods=["POST", "GET"])
def index():
    if 'loggedin' in session:
        data = LogoDB.show_logo(self=LogoDB)
        data_model = LogoDB.get_all_model(self=LogoDB)
        data_user = LogoDB.get_all_user(self=LogoDB)
        datavideo = []
        if session['user_role'] == int(0):
            datavideo = LogoDB.get_video_by_userid(
                self=LogoDB, user_id=session['user_id'])
        else:
            datavideo = LogoDB.get_all_video(self=LogoDB)
        return render_template("index.html", data=data, data_model=data_model, datavideo=datavideo, data_user=data_user, sessionnya=session)
    return redirect(url_for('login'))


@app.route("/", methods=["GET", "POST"])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        account = LogoDB.login_check(
            self=LogoDB, username=username, password=password)
        print(account)
        if account:
            session['loggedin'] = True
            session['user_id'] = account['user_id']
            session['user_name'] = account['user_name']
            session['user_email'] = account['user_email']
            session['user_namalengkap'] = account['user_namalengkap']
            session['user_role'] = account['user_role']
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('user_email', None)
    session.pop('user_namalengkap', None)
    return redirect(url_for('login'))


@app.route("/<logo_id>/detail", methods=["GET"])
def detail(logo_id):
    data = LogoDB.get_one_logo(self=LogoDB, logo_id=logo_id)
    print(data['logo_nama'])
    filelogoandpath = []

    splitpath = data['logo_path'].split('/')
    forpath = splitpath[1] + '/' + splitpath[2]+'/'
    filelogo = os.listdir(data['logo_path'])
    for i in filelogo:
        newpathfile = forpath + i
        filelogoandpath.append(newpathfile)

    return render_template("gallery.html", data=data, filelogo=filelogoandpath)


@app.route("/generate", methods=["POST", "GET"])
def generate():
    if 'loggedin' in session:
        return render_template("generate.html", sessionnya=session)
    return redirect(url_for('login'))


@app.route("/create_video", methods=["POST", "GET"])
def create_video():
    if 'loggedin' in session:
        data = LogoDB.get_all_model(self=LogoDB)
        return render_template("upload_video.html", data=data, sessionnya=session)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    start = time.time()
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'static/uploads', secure_filename(f.filename))
                fname_save = request.form.get('nama_file', type=str)
                path_save = 'static/results/' + fname_save + \
                    '_' + str(round(time.time())) + '/'
                isExist = os.path.exists(path_save)
                f.save(file_path)

                logo_nama = request.form.get('nama_file', type=str)
                logo_filename = filename
                logo_path = path_save
                hasil_add = LogoDB.add_logo(
                    self=LogoDB, logo_nama=logo_nama, logo_filename=logo_filename, logo_path=logo_path)
                if(hasil_add == 'berhasil'):
                    if not isExist:
                        os.makedirs(path_save)
                        try:
                            run(GenerateData(), file_path, path_save, fname_save)
                        except Exception as e:
                            print(e)
                    timing = time.time() - start
                    trans_path = path_save + fname_save + '_trans_' + \
                        str(random.randint(5, 50)) + '.png'
                    bw_path = path_save + 'data_' + fname_save + \
                        '_bw_' + str(random.randint(5, 50)) + '.png'
                    rs_path = path_save + 'data_' + fname_save + \
                        '_resized_' + str(random.randint(5, 50)) + '.png'
                    rsheight_path = path_save + 'data_' + fname_save + \
                        '_resized_height_' + \
                        str(random.randint(40, 95)) + '.png'
                    rswidth_path = path_save + 'data_' + fname_save + \
                        '_resized_width_' + \
                        str(random.randint(40, 95)) + '.png'
                    rt_path = path_save + 'data_' + fname_save + \
                        '_rotated_' + str(random.randint(5, 50)) + '.png'
                    wb_path = path_save + 'data_' + fname_save + \
                        '_wb_' + str(random.randint(5, 50)) + '.png'

                    return jsonify({'timing': timing, 'filename': fname_save, 'trans_path': trans_path, 'bw_path': bw_path, 'rs_path': rs_path, 'rsheight_path': rsheight_path, 'rswidth_path': rswidth_path, 'rt_path': rt_path, 'wb_path': wb_path, 'htmlresponse': render_template('index.html', trans_path=trans_path, bw_path=bw_path, rs_path=rs_path, rsheight_path=rsheight_path, rswidth_path=rswidth_path, rt_path=rt_path, wb_path=wb_path)})
                else:
                    print(hasil_add)
                # return jsonify({'timing':timing, 'filename':fname_save, 'bw_path':bw_path})

        else:
            flash('No selected file')
            return redirect(request.url)


@app.route('/detect', methods=['POST', 'GET'])
def detection():
    start = time.time()
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                basepath = os.path.dirname(__file__)
                file_path = os.path.join(
                    basepath, 'static/input/', secure_filename(f.filename))
                # return jsonify({'res':f'{temp_f}.jpg'})
                f.save(file_path)
                timing = time.time() - start
                # # generate = gen('static/uploads/'+filename,filename)
                # if(generate == True):
                #     return jsonify({'filename':str(filename)})
                model_id = request.form.getlist('model_id')
                ads_value = request.form.get('ads_value', type=str)
                ads = ads_value.replace(",", "")
                filename_awal = str(filename)
                for i in model_id:
                    filename_akhir = str(filename.rsplit(
                        '.', 1)[0]) + '_' + str(time.time()) + '.mp4'
                    LogoDB.add_video(self=LogoDB, arr_video=[
                                     filename_awal, filename_akhir, i, ads, session['user_id']])
                    video = LogoDB.get_one_video(
                        self=LogoDB, video_id=filename_akhir)
                    video_id = filename_akhir
                    model = video['model_id']
                    # print(len(model_id))
                    yolo = YoloDetector()
                    yolomain = yolo.main(
                        id_model=model, video_id=video_id, thresh=0.5, ads=ads)
                if(yolomain):
                    timing = time.time() - start
                    datanya = LogoDB.get_output(self=LogoDB, video_id=video_id)
                    return jsonify({'filename': str(filename_akhir), 'timing': timing, 'datanya': datanya, 'video_id': video['video_id']})
                else:
                    print('gagal')
        else:
            flash('No selected file')
            return redirect(request.url)

@app.route('/add_users', methods=['POST'])
def create_user():
    if request.method == 'POST':
        user_name = request.form.get('user_name', type=str)
        user_password = request.form.get('user_password', type=str)
        user_namalengkap = request.form.get('user_namalengkap', type=str)
        user_role = request.form.get('user_role', type=str)
        LogoDB.add_user(self=LogoDB, arr_user=[
                            user_name, user_password, user_namalengkap, user_role])
        return jsonify({'status': True, 'message': 'Data berhasil ditambahkan!'})
    else:
        flash('No selected file')
        return redirect(url_for('index'))

@app.route('/edit_users', methods=['POST'])
def update_user():
    if request.method == 'POST':
        user_name = request.form.get('user_name', type=str)
        user_password = request.form.get('user_password', type=str)
        user_namalengkap = request.form.get('user_namalengkap', type=str)
        user_role = request.form.get('user_role', type=str)
        user_id = request.form.get('user_id', type=str)
        arr_user=[user_name, user_namalengkap, user_password, user_role, user_id]
        print(arr_user)
        LogoDB.update_user(self=LogoDB, arr_user=[user_name, user_password, user_namalengkap, user_role, user_id])
        return jsonify({'status': True, 'message': 'Data berhasil diubah!'})
    else:
        flash('No selected file')
        return redirect(url_for('index'))

@app.route('/add_model', methods=['POST'])
def add_model():
    if request.method == 'POST':
        f = request.files['model_weights']
        f2 = request.files['model_config']
        f3 = request.files['model_label']
        basepath = os.path.dirname(__file__)
        model_name = request.form.get('model_name', type=str)
        file_path = os.path.join(
            basepath, 'static/models/', model_name)
        isExist = os.path.exists(file_path)
        if not isExist:
            os.mkdir(file_path)
            path1 = os.path.join(file_path, f.filename)
            path2 = os.path.join(file_path, f2.filename)
            path3 = os.path.join(file_path, f3.filename)
            f.save(path1)
            f2.save(path2)
            f3.save(path3)
            LogoDB.add_model(self=LogoDB, arr_model=[
                             model_name, f.filename, f2.filename, f3.filename])
            # print(data)
            return jsonify({'status': True, 'message': 'Data berhasil ditambahkan!'})
        else:
            return jsonify({'error': True, 'message': 'Data sudah tersedia!'})

        # os.mkdir(file_path)
    else:
        flash('No selected file')
        return redirect(url_for('index'))


@app.route('/hapus_model', methods=['POST'])
def delete_model():
    if request.method == 'POST':
        id_model = request.form.get('id', type=str)
        data = LogoDB.get_one_model(self=LogoDB, model_id=id_model)
        if (data):
            isExist = os.path.exists(file_path)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static/models/', data['model_nama'])
            if isExist:
                os.remove(file_path)
                # print(data)
                LogoDB.delete_model(self=LogoDB, id_model=id_model)
                return jsonify({'status': True, 'message': 'Data berhasil dihapus!'})
            else:
                return jsonify({'error': True, 'message': 'Data gagal terhapus!'})
        else:
            return jsonify({'error': True, 'message': 'Data gagal terhapus!'})
    else:
        flash('No selected file')
        return redirect(url_for('index'))
        
@app.route('/hapus_user', methods=['POST'])
def delete_user():
    if request.method == 'POST':
        id_user = request.form.get('id', type=str)
        data = LogoDB.get_one_user(self=LogoDB, user_id=id_user)
        if (data):
            LogoDB.delete_user(self=LogoDB, id_user=id_user)
            return jsonify({'status': True, 'message': 'Data berhasil dihapus!'})
        else:
            return jsonify({'error': True, 'message': 'Data gagal terhapus!'})
    else:
        flash('No selected file')
        return redirect(url_for('index'))

# @app.route('/detect_gambar', methods=['POST', 'GET'])
# def detection_gambar():
#     start = time.time()
#     if request.method == 'POST':
#         if 'file' in request.files:
#             f = request.files['file']
#             if f.filename == '':
#                 flash('No selected file')
#                 return redirect(request.url)
#             if f and allowed_file(f.filename):
#                 model = get_conv_model()
#                 file = f.read()
#                 np_img = np.fromstring(file, np.uint8)
#                 img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#                 img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
#                 temp_f = f.filename[::-1]
#                 temp_f = f"{temp_f[temp_f.find('.')+1:][::-1]}"
#                 new_path = f"static/temp/{temp_f}.jpg"
#                 cv2.imwrite(new_path, img)
#                 dim = (224, 224)
#                 prob, cat, new_img, lime_img = get_img_prediction_bounding_box(new_path, dim, model)
#                 cv2.imwrite(new_path, new_img)
#                 cv2.imwrite(f"static/temp/{temp_f}Lime.jpg", lime_img)
#                 # return jsonify({'res':f'{temp_f}.jpg'}) 
#                 timing = time.time() - start
#                 return jsonify({'res':f'static/temp/{temp_f}.jpg', 'timing':timing})
                
#         else:
#             flash('No selected file')
#             return redirect(request.url)
@app.route('/video_upload/<filepath>')
def video_upload(filepath):
    video = Response(gen('static/uploads/'+filepath, filepath),
                     mimetype='multipart/x-mixed-replace; boundary=frame')
    FILENAMES = filepath
    # print('ada nih')
    # return send_from_directory(
    #     app.config['UPLOAD_FOLDER'], filepath, as_attachment=True
    # )
    # print(video)
    return video, FILENAMES


@app.route('/display_video/<filename>')
def display_video(filename):
    return Response(gen_frames('static/output/'+filename), mimetype='multipart/x-mixed-replace; boundary=frame')
    # filename = 'outputs.mp4'
    # return redirect(url_for('static', filename='results/' + filename), code=301)


@app.route('/export/<video_id>')
def export(video_id):
    video = LogoDB.get_video_by_videoid(self=LogoDB, video_id=video_id)
    result = LogoDB.get_output_by_videoid(self=LogoDB, video_id=video_id)
    output = io.BytesIO()
    filename = video['video_filename_akhir']
    # create WorkBook object
    workbook = xlwt.Workbook()
    # add a sheet
    sh = workbook.add_sheet('Generate Results')

    # add headers
    sh.write(0, 0, 'No')
    sh.write(0, 1, 'Logo Name')
    sh.write(0, 2, 'Start (M:S)')
    sh.write(0, 3, 'End (M:S')
    sh.write(0, 4, 'Duration (Seconds)')
    sh.write(0, 5, 'Ads Value')

    idx = 0
    for row in result:
        sh.write(idx+1, 0, str(row['num']))
        sh.write(idx+1, 1, row['model_nama'])
        sh.write(idx+1, 2, row['start_time'])
        sh.write(idx+1, 3, row['end_time'])
        sh.write(idx+1, 4, row['durasi_sec'])
        sh.write(idx+1, 5, row['ads_per_menit'])
        idx += 1

    workbook.save(output)
    output.seek(0)

    return Response(output, mimetype="application/ms-excel", headers={"Content-Disposition": "attachment;filename=" + filename + ".xls"})


@app.route('/detail_video/<video_id>')
def detail_video(video_id):
    if 'loggedin' in session:
        video = LogoDB.get_video_by_videoid(self=LogoDB, video_id=video_id)
        outputs = LogoDB.get_output_by_videoid(self=LogoDB, video_id=video_id)
        return render_template('detail_video.html', video=video, outputs=outputs, sessionnya=session)
    else:
        return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
