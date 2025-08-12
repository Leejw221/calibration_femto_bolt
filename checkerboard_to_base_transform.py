import numpy as np

# 카메라에서 체커보드로 가는 변환행렬
T_camera_to_checkerboard = np.array([
    [-0.994138,  0.003833,  0.026560,  215.820847],
    [ 0.007131, -0.914910,  0.403594, -168.838913],
    [ 0.025717,  0.403641,  0.914552, -502.208710],
    [ 0.000000,  0.000000,  0.000000,    1.000000]
])

# 체커보드에서 base로 가는 변환행렬  
T_checkerboard_to_base = np.array([
    [ 0,  1,  0,  220],
    [ 1,  0,  0, -225],
    [ 0,  0, -1,    6],
    [ 0,  0,  0,    1]
])

print("=== 주어진 변환행렬들 ===")
print("카메라→체커보드:")
np.set_printoptions(precision=6, suppress=True)
print(T_camera_to_checkerboard)
print()
print("체커보드→base:")
print(T_checkerboard_to_base)
print()

# 카메라에서 base로 가는 변환행렬 계산
# T_camera_to_base = T_checkerboard_to_base × T_camera_to_checkerboard
T_camera_to_base = np.dot(T_checkerboard_to_base, T_camera_to_checkerboard)

print("=== 최종 결과 ===")
print("카메라→base 변환행렬:")
print(T_camera_to_base)
