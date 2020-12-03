import numpy as np
import cv2
import glob
from os.path import join
import json
from data_utility import image_normalization
import os
import csv
import tensorflow as tf
import global_variables as var
import random

dataset_path = var.GazeCapture_path


# GazeCaptureを取り扱うクラス及び関数群
class getGazeCapture:

    # データセットから先頭のx名分の画像を指定したパスに保存する関数
    def saveGazeCapture(self, smallData_path, save_path, model_path, amount):
        self.save_path = save_path
        self.model_path = model_path

        os.makedirs(save_path, exist_ok=True)

        # トリミングする画像のサイズ指定
        img_cols = 128
        img_rows = 128
        img_ch = 3

        # 推定値を共に併記するためのモデル
        model = tf.keras.models.load_model(self.model_path)

        # saveTrimedFace(smallData_path, save_path, amount, img_ch, img_cols, img_rows, model)
        saveClippedImgRandom(smallData_path, save_path, amount, img_ch, img_cols, img_rows, model)


# load a batch of random data given the full list of the dataset
def saveTrimedFace(smallData_path, save_path, batch_size, img_ch, img_cols, img_rows, model):
    save_img = True

    # data structures for batches
    left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 25, 25, 1), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)
    y_batch_test = np.zeros((2,), dtype=np.float32)

    # 一人ずつのリスト
    seq_list = []
    seqs = sorted(glob.glob(join(smallData_path, "0*")))

    # 人の数を表現している
    for b, seq in enumerate(seqs):

        print("wwwww")

        seq_list = []
        file = open(seq, "r")
        content = file.read().splitlines()

        for line in content:
            seq_list.append(line)

        os.makedirs(save_path + seq_list[0][:5], exist_ok=True)

        # 人ごとのディレクトリ
        personDir = save_path + seq_list[0][:5]

        for person in seq_list:

            print("bbbbb")
            # get the lucky one
            img_name = person

            # directory
            dir = img_name[:5]

            # frame name
            frame = img_name[6:]

            # frame
            frame_name = img_name[6:11]

            # index of the frame into a sequence
            idx = int(frame[:-4])

            # open json files
            face_file = open(join(dataset_path, dir, "appleFace.json"))
            left_file = open(join(dataset_path, dir, "appleLeftEye.json"))
            right_file = open(join(dataset_path, dir, "appleRightEye.json"))
            dot_file = open(join(dataset_path, dir, "dotInfo.json"))
            grid_file = open(join(dataset_path, dir, "faceGrid.json"))

            # load json content
            face_json = json.load(face_file)
            left_json = json.load(left_file)
            right_json = json.load(right_file)
            dot_json = json.load(dot_file)
            grid_json = json.load(grid_file)

            # open image
            img = cv2.imread(join(dataset_path, dir, "frames", frame))

            # if image is null, skip
            if img is None:
                # print("Error opening image: {}".format(join(path, dir, "frames", frame)))
                continue

            # if coordinates are negatives, skip (a lot of negative coords!)
            if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
                    int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
                    int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
                # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
                continue

            # get face
            tl_x_face = int(face_json["X"][idx])
            tl_y_face = int(face_json["Y"][idx])
            w = int(face_json["W"][idx])
            h = int(face_json["H"][idx])
            br_x = tl_x_face + w
            br_y = tl_y_face + h
            face = img[tl_y_face:br_y, tl_x_face:br_x]

            # get left eye
            tl_x = tl_x_face + int(left_json["X"][idx])
            tl_y = tl_y_face + int(left_json["Y"][idx])
            w = int(left_json["W"][idx])
            h = int(left_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            left_eye = img[tl_y:br_y, tl_x:br_x]

            # get right eye
            tl_x = tl_x_face + int(right_json["X"][idx])
            tl_y = tl_y_face + int(right_json["Y"][idx])
            w = int(right_json["W"][idx])
            h = int(right_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            right_eye = img[tl_y:br_y, tl_x:br_x]

            # get face grid (in ch, cols, rows convention)
            face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
            tl_x = int(grid_json["X"][idx])
            tl_y = int(grid_json["Y"][idx])
            w = int(grid_json["W"][idx])
            h = int(grid_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            face_grid[tl_y:br_y, tl_x:br_x, 0] = 1

            # get labels
            y_x = dot_json["XCam"][idx]
            y_y = dot_json["YCam"][idx]
            y_batch_test[0] = y_x
            y_batch_test[1] = y_y

            # resize images
            face = cv2.resize(face, (img_cols, img_rows))
            left_eye = cv2.resize(left_eye, (img_cols, img_rows))
            right_eye = cv2.resize(right_eye, (img_cols, img_rows))

            grid_width = 25
            grid_height = 25

            gridImg = np.zeros((grid_height, grid_width, 3), np.uint8)

            gridImg[tl_y:br_y, tl_x:br_x] = [255, 255, 255]

            os.makedirs(personDir + "/" + frame_name, exist_ok=True)
            f = open(personDir + "/data.csv", "a")
            csvWriter = csv.writer(f)
            # save images (for debug)
            if save_img:
                cv2.imwrite(personDir + "/" + frame_name + "/face.png", face)
                cv2.imwrite(personDir + "/" + frame_name + "/right.png", right_eye)
                cv2.imwrite(personDir + "/" + frame_name + "/left.png", left_eye)
                cv2.imwrite(personDir + "/" + frame_name + "/image.png", img)
                cv2.imwrite(personDir + "/" + frame_name + "/grid.png", gridImg)

                listData = []
                listData.append(y_x)
                listData.append(y_y)

                csvWriter.writerow(listData)

                # normalization
                face = image_normalization(face)
                left_eye = image_normalization(left_eye)
                right_eye = image_normalization(right_eye)

                left = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                right = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                facebatch = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                grid = np.zeros(shape=(1, 25, 25, 1), dtype=np.float32)

                right[0] = right_eye
                left[0] = left_eye
                grid[0] = face_grid
                facebatch[0] = face

                x = [right, left, facebatch, grid]
                estimate = model.predict(x)
                estimateValue = [estimate[0, 0], estimate[0, 1]]
                csvWriter.writerow(estimateValue)

            ######################################################

            # transpose images
            # face = face.transpose(2, 0, 1)
            # left_eye = left_eye.transpose(2, 0, 1)
            # right_eye = right_eye.transpose(2, 0, 1)

            # check data types
            face = face.astype('float32')
            left_eye = left_eye.astype('float32')
            right_eye = right_eye.astype('float32')

            # add to the related batch
            left_eye_batch[b] = left_eye
            right_eye_batch[b] = right_eye
            face_batch[b] = face
            face_grid_batch[b] = face_grid
            y_batch[b][0] = y_x
            y_batch[b][1] = y_y

        if b == batch_size:
            break

    x = [right_eye_batch, left_eye_batch, face_batch, face_grid_batch]
    predictions = model.predict(x)

    for i, prediction in enumerate(predictions):
        f = open(save_path + str(i) + "/data.csv", "a")
        csvWriter = csv.writer(f)

        listData = []
        listData.append(prediction[0])
        listData.append(prediction[1])
        csvWriter.writerow(listData)

    # print and analyze predictions
    err_x = []
    err_y = []
    for i, prediction in enumerate(predictions):
        print("PR: {} {}".format(prediction[0], prediction[1]))
        print("GT: {} {} \n".format(y_batch[i][0], y_batch[i][1]))

        err_x.append(abs(prediction[0] - y_batch[i][0]))
        err_y.append(abs(prediction[1] - y_batch[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ({} samples)".format(mae_x, mae_y, len(y_batch)))
    print("STD: {} {} ({} samples)".format(std_x, std_y, len(y_batch)))

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# load a batch of random data given the full list of the dataset
def saveClippedImgRandom(smallData_path, save_path, batch_size, img_ch, img_cols, img_rows, model):
    save_img = True

    # data structures for batches
    left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 25, 25, 1), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)
    y_batch_test = np.zeros((2,), dtype=np.float32)

    num = 4
    count = 0

    # 一人ずつのリスト
    seq_list = []
    seqs = sorted(glob.glob(join(smallData_path, "0*")))

    # 人の数を表現している
    for b, seq in enumerate(seqs):

        print("wwwww")

        seq_list = []
        file = open(seq, "r")
        content = file.read().splitlines()

        for line in content:
            seq_list.append(line)

        #os.makedirs(save_path + seq_list[0][:5], exist_ok=True)

        person_count = 0

        for i in range(100):

            # for person in seq_list:
            if len(seq_list) == 0:
                break
            person = random.choice(seq_list)

            print("bbbbb")
            # get the lucky one
            img_name = person

            # directory
            dir = img_name[:5]

            # frame name
            frame = img_name[6:]

            # frame
            frame_name = img_name[6:11]

            # index of the frame into a sequence
            idx = int(frame[:-4])

            # open json files
            face_file = open(join(dataset_path, dir, "appleFace.json"))
            left_file = open(join(dataset_path, dir, "appleLeftEye.json"))
            right_file = open(join(dataset_path, dir, "appleRightEye.json"))
            dot_file = open(join(dataset_path, dir, "dotInfo.json"))
            grid_file = open(join(dataset_path, dir, "faceGrid.json"))

            # load json content
            face_json = json.load(face_file)
            left_json = json.load(left_file)
            right_json = json.load(right_file)
            dot_json = json.load(dot_file)
            grid_json = json.load(grid_file)

            # open image
            img = cv2.imread(join(dataset_path, dir, "frames", frame))

            # if image is null, skip
            if img is None:
                # print("Error opening image: {}".format(join(path, dir, "frames", frame)))
                continue

            # if coordinates are negatives, skip (a lot of negative coords!)
            if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
                    int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
                    int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
                # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
                continue

            # get face
            tl_x_face = int(face_json["X"][idx])
            tl_y_face = int(face_json["Y"][idx])
            w = int(face_json["W"][idx])
            h = int(face_json["H"][idx])
            br_x = tl_x_face + w
            br_y = tl_y_face + h
            face = img[tl_y_face:br_y, tl_x_face:br_x]

            # get left eye
            tl_x = tl_x_face + int(left_json["X"][idx])
            tl_y = tl_y_face + int(left_json["Y"][idx])
            w = int(left_json["W"][idx])
            h = int(left_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            left_eye = img[tl_y:br_y, tl_x:br_x]

            # get right eye
            tl_x = tl_x_face + int(right_json["X"][idx])
            tl_y = tl_y_face + int(right_json["Y"][idx])
            w = int(right_json["W"][idx])
            h = int(right_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            right_eye = img[tl_y:br_y, tl_x:br_x]

            # get face grid (in ch, cols, rows convention)
            face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
            tl_x = int(grid_json["X"][idx])
            tl_y = int(grid_json["Y"][idx])
            w = int(grid_json["W"][idx])
            h = int(grid_json["H"][idx])
            br_x = tl_x + w
            br_y = tl_y + h
            face_grid[tl_y:br_y, tl_x:br_x, 0] = 1

            # get labels
            y_x = dot_json["XCam"][idx]
            y_y = dot_json["YCam"][idx]
            y_batch_test[0] = y_x
            y_batch_test[1] = y_y

            # resize images
            face = cv2.resize(face, (img_cols, img_rows))
            left_eye = cv2.resize(left_eye, (img_cols, img_rows))
            right_eye = cv2.resize(right_eye, (img_cols, img_rows))

            grid_width = 25
            grid_height = 25

            gridImg = np.zeros((grid_height, grid_width, 3), np.uint8)

            gridImg[tl_y:br_y, tl_x:br_x] = [255, 255, 255]

            # os.makedirs(personDir + "/" + frame_name, exist_ok=True)
            f = open(save_path + "/data.csv", "a")
            csvWriter = csv.writer(f)
            # save images (for debug)
            if save_img:
                cv2.imwrite(save_path + "/face/face_{}.png".format(count), face)
                cv2.imwrite(save_path + "/right_eye/right_{}.png".format(count), right_eye)
                cv2.imwrite(save_path + "/left_eye/left_{}.png".format(count), left_eye)
                cv2.imwrite(save_path + "/image/image_{}.png".format(count), img)
                cv2.imwrite(save_path + "/grid/grid_{}.png".format(count), gridImg)

                # listData = []
                # listData.append(y_x)
                # listData.append(y_y)

                # csvWriter.writerow(listData)

                # normalization
                face = image_normalization(face)
                left_eye = image_normalization(left_eye)
                right_eye = image_normalization(right_eye)

                left = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                right = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                facebatch = np.zeros(shape=(1, img_cols, img_rows, img_ch), dtype=np.float32)
                grid = np.zeros(shape=(1, 25, 25, 1), dtype=np.float32)

                right[0] = right_eye
                left[0] = left_eye
                grid[0] = face_grid
                facebatch[0] = face

                x = [right, left, facebatch, grid]
                estimate = model.predict(x)
                estimateValue = [y_x, y_y, estimate[0, 0], estimate[0, 1]]
                csvWriter.writerow(estimateValue)

            count = count + 1
            person_count = person_count + 1
            if person_count == 4:
                break





# こいつは廃棄物だけど一時保管
# load a batch of random data given the full list of the dataset
"""
def saveTrimedFace(names, path, batch_size, img_ch, img_cols, img_rows, model):

    save_img = True

    # data structures for batches
    left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 25, 25, 1), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)
    y_batch_test = np.zeros((2,), dtype=np.float32)

    # counter for check the size of loading batch
    b = 0
    
    # 一人ずつのリスト
    seq_list = []
    seqs = sorted(glob.glob(join(path, "0*")))
    
    # 人の数を表現している
    for b, seq in enumurate(seqs):

        seq_list = []
        file = open(seq, "r")
        content = file.read().splitlines()
        
        for line in content:
            seq_list.append(line)
            
        for frame in seq_list:
        # lottery
        i = np.random.randint(0, len(names))

        # get the lucky one
        img_name = frame

        # directory
        dir = img_name[:5]

        # frame name
        frame = img_name[6:]

        # index of the frame into a sequence
        idx = int(frame[:-4])

        # open json files
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))

        # load json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, dir, "frames", frame))

        # if image is null, skip
        if img is None:
            # print("Error opening image: {}".format(join(path, dir, "frames", frame)))
            continue

        # if coordinates are negatives, skip (a lot of negative coords!)
        if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
            int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
            int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
            # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[tl_y:br_y, tl_x:br_x, 0] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]
        y_batch_test[0] = y_x
        y_batch_test[1] = y_y

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        grid_width = 25
        grid_height = 25

        gridImg = np.zeros((grid_height, grid_width, 3), np.uint8)

        gridImg[tl_y:br_y, tl_x:br_x] = [255,255,255]

        os.makedirs(save_path + str(b), exist_ok=True)
        f = open(save_path + str(b) + "/data.csv", "a")
        csvWriter = csv.writer(f)
        # save images (for debug)
        if save_img:
            cv2.imwrite(save_path + str(b) + "/face.png", face)
            cv2.imwrite(save_path + str(b) + "/right.png", right_eye)
            cv2.imwrite(save_path + str(b) + "/left.png", left_eye)
            cv2.imwrite(save_path + str(b) + "/image.png", img)
            cv2.imwrite(save_path + str(b) + "/grid.png", gridImg)

            listData = []
            listData.append(y_x)
            listData.append(y_y)
            csvWriter.writerow(listData)

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        # face = face.transpose(2, 0, 1)
        # left_eye = left_eye.transpose(2, 0, 1)
        # right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')


        # add to the related batch
        left_eye_batch[b] = left_eye
        right_eye_batch[b] = right_eye
        face_batch[b] = face
        face_grid_batch[b] = face_grid
        y_batch[b][0] = y_x
        y_batch[b][1] = y_y
        
        if b == batch_size:
            break

    x = [right_eye_batch, left_eye_batch, face_batch, face_grid_batch]
    predictions = model.predict(x)
    
    for i,prediction in enumerate(predictions):

        f = open(save_path + str(i) + "/data.csv", "a")
        csvWriter = csv.writer(f)
        
        listData = []
        listData.append(prediction[0])
        listData.append(prediction[1])
        csvWriter.writerow(listData)

    # print and analyze predictions
    err_x = []
    err_y = []
    for i, prediction in enumerate(predictions):
        print("PR: {} {}".format(prediction[0], prediction[1]))
        print("GT: {} {} \n".format(y_batch[i][0], y_batch[i][1]))

        err_x.append(abs(prediction[0] - y_batch[i][0]))
        err_y.append(abs(prediction[1] - y_batch[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ({} samples)".format(mae_x, mae_y, len(y_batch)))
    print("STD: {} {} ({} samples)".format(std_x, std_y, len(y_batch)))

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch

"""
