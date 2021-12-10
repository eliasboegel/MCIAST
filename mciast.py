from ctypes import c_void_p, c_double, c_int, cdll

mciast_lib = cdll.LoadLibrary("./mciast.so")


#spreading_pressure_langmuir = lib.spreading_pressure_langmuir
#spreading_pressure_langmuir.restype = c_double

