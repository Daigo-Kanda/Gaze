import getGazeCapture as gc
import global_variables as var

save_path = "/mnt/data2/img/20200209/"
smallData_path = "/home/daigokanda/GazeCapture_DataSet/smallSet/train"
gazeCapture = gc.getGazeCapture()
gazeCapture.saveGazeCapture(smallData_path, save_path, var.model_path, 50)
