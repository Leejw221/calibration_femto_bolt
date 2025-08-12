#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 간단한 방법: RGB->체커보드 변환행렬 구한 후 depth->RGB 변환행렬 곱하기
"""
import sys
import cv2
import numpy as np
import pyorbbecsdk as ob
from utils import frame_to_bgr_image

class FinalDepthCalibration:
    def __init__(self):
        self.checkerboard_size = (9, 6)  # 9x6 교차점 (10x7 사각형)
        self.square_size = 55.0  # mm - 실제 측정값
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        
        # 체커보드 원점 설정 옵션
        # 🔧 OpenCV findChessboardCorners는 항상 top_left 순서로 검출합니다
        # 실제 물리적 배치와 관계없이 이미지상에서 왼쪽 위부터 검출
        self.origin_corner = "top_left"  # OpenCV 기본 검출 순서에 맞춤
        print(f"체커보드 설정: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 교차점")
        print(f"사각형 크기: {self.square_size}mm")
        print(f"원점 위치: {self.origin_corner} (OpenCV 검출 순서)")
        
    def setup_camera(self):
        try:
            # RGB + Depth 센서 활성화
            color_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            depth_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            
            # 🔧 시각화와 동일한 해상도로 설정 (일관성을 위해)
            # Color: 1280x720, Depth: 320x288 (시각화와 동일)
            color_profile = None
            depth_profile = None
            
            try:
                # 시각화와 동일한 해상도 시도
                color_profile = color_profiles.get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
                depth_profile = depth_profiles.get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
                print("✓ 시각화와 동일한 해상도 설정")
            except:
                # 실패시 기본 프로파일 사용
                color_profile = color_profiles.get_default_video_stream_profile()
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print("⚠ 기본 해상도로 fallback")
            
            # 해상도 확인
            cp = color_profile.as_video_stream_profile()
            dp = depth_profile.as_video_stream_profile()
            print(f"✓ Color 해상도: {cp.get_width()}x{cp.get_height()}")
            print(f"✓ Depth 해상도: {dp.get_width()}x{dp.get_height()}")
            
            self.config.enable_stream(color_profile)
            self.config.enable_stream(depth_profile)
            
            self.pipeline.start(self.config)
            return True
        except Exception as e:
            print(f"카메라 설정 실패: {e}")
            return False
    
    def create_checkerboard_3d_points(self):
        """체커보드 3D 좌표 생성 - 원점 위치에 따라 조정"""
        num_points = self.checkerboard_size[0] * self.checkerboard_size[1]
        objp = np.zeros((num_points, 3), np.float32)
        
        # 기본 격자 생성 (top_left 기준)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0],
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # mm 단위로 스케일링
        
        # 원점 위치에 따른 좌표 변환
        if self.origin_corner == "bottom_right":
            # X축 뒤집기 + Y축 뒤집기
            objp[:, 0] = (self.checkerboard_size[0] - 1) * self.square_size - objp[:, 0]
            objp[:, 1] = (self.checkerboard_size[1] - 1) * self.square_size - objp[:, 1]
        elif self.origin_corner == "top_right":
            # X축만 뒤집기
            objp[:, 0] = (self.checkerboard_size[0] - 1) * self.square_size - objp[:, 0]
        elif self.origin_corner == "bottom_left":
            # Y축만 뒤집기
            objp[:, 1] = (self.checkerboard_size[1] - 1) * self.square_size - objp[:, 1]
        # top_left는 그대로 사용 (OpenCV 기본)
        
        return objp
        
    def capture_checkerboard(self):
        print(f"\n체커보드 캘리브레이션 시작")
        print(f"- 패턴: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 교차점")
        print(f"- 크기: {self.square_size}mm 사각형")
        print(f"- 원점: {self.origin_corner} (맨아래, 맨오른쪽)")
        print("- 📍 중요: OpenCV는 체커보드를 검출한 첫 번째 코너부터")
        print("          순서대로 좌표를 할당합니다. (보통 맨아래 맨오른쪽부터)")
        print("- 🔧 카메라가 약간 기울어져 있어도 체커보드를 수평으로 배치하세요")
        print("- 체커보드가 인식되면 SPACE 키를 누르세요")
        print("- 종료하려면 'q' 키를 누르세요")
        
        # 🔧 체커보드 검출 개선 설정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)  # 더 정밀한 기준
        
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                continue
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # RGB 이미지 변환
            try:
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue
            except Exception as e:
                print(f"이미지 변환 오류: {e}")
                continue
                
            # 체커보드 검출 (개선된 방법)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # 🔧 이미지 전처리 추가 (대비 향상)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 가우시안 블러로 노이즈 제거
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 체커보드 찾기 - 다양한 플래그 시도
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
            
            # 표시
            display = color_image.copy()
            if ret:
                # 🔧 서브픽셀 정확도로 코너 개선 (더 정밀한 설정)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 체커보드 그리기
                cv2.drawChessboardCorners(display, self.checkerboard_size, corners, ret)
                
                # 📊 체커보드 품질 평가
                quality_score = self.evaluate_checkerboard_quality(corners, gray.shape)
                
                cv2.putText(display, f"Found {len(corners)} corners! Press SPACE", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Quality: {quality_score:.1f}% | Origin: {self.origin_corner}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"Keep checkerboard FLAT on desk", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(display, f"No checkerboard ({self.checkerboard_size[0]}x{self.checkerboard_size[1]})", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, f"Square size: {self.square_size}mm", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display, f"Keep checkerboard FLAT on desk", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.namedWindow('Final Calibration', cv2.WINDOW_NORMAL)
            cv2.imshow('Final Calibration', display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and ret:
                # 품질 체크
                if quality_score < 70:
                    print(f"⚠ 체커보드 품질이 낮습니다 ({quality_score:.1f}%). 더 나은 각도로 다시 시도하세요.")
                    continue
                    
                print(f"\n캘리브레이션 시작... (검출된 코너: {len(corners)}개, 품질: {quality_score:.1f}%)")
                success = self.calculate_final_transform(color_frame, depth_frame, corners)
                if success:
                    print("✓ 캘리브레이션 완료!")
                    break
                else:
                    print("✗ 캘리브레이션 실패. 다시 시도하세요.")
            elif key == ord('q'):
                return False
                
        cv2.destroyAllWindows()
        return True
    
    def evaluate_checkerboard_quality(self, corners, image_shape):
        """체커보드 검출 품질 평가"""
        if corners is None or len(corners) == 0:
            return 0.0
            
        corners = corners.reshape(-1, 2)
        
        # 1. 코너들이 이미지 경계에서 충분히 떨어져 있는지
        h, w = image_shape
        margin = min(w, h) * 0.1  # 10% 마진
        
        edge_penalty = 0
        for corner in corners:
            x, y = corner
            if x < margin or x > w - margin or y < margin or y > h - margin:
                edge_penalty += 1
                
        edge_score = max(0, 100 - (edge_penalty / len(corners)) * 100)
        
        # 2. 코너들의 분포가 고른지 (전체 이미지 영역 사용)
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        
        x_range = (x_coords.max() - x_coords.min()) / w
        y_range = (y_coords.max() - y_coords.min()) / h
        
        coverage_score = min(100, (x_range + y_range) * 50)
        
        # 3. 격자 정렬 품질 (직선성)
        grid_h, grid_w = self.checkerboard_size
        alignment_score = 100  # 기본값
        
        try:
            # 첫 번째 행의 점들이 직선인지 확인 (numpy만 사용)
            first_row = corners[:grid_w]
            if len(first_row) >= 3:
                # 단순 선형성 체크 (분산 기반)
                x_coords = first_row[:, 0]
                y_coords = first_row[:, 1]
                
                # 점들이 직선에 얼마나 가까운지 계산
                x_var = np.var(x_coords)
                y_var = np.var(y_coords)
                
                if x_var > 0 and y_var > 0:
                    # 상관계수로 직선성 평가
                    correlation = np.corrcoef(x_coords, y_coords)[0, 1]
                    alignment_score = min(100, abs(correlation) * 100)
        except:
            pass
            
        # 최종 점수 (가중 평균)
        final_score = (edge_score * 0.3 + coverage_score * 0.4 + alignment_score * 0.3)
        
        return final_score
    
    def calculate_final_transform(self, color_frame, depth_frame, corners_2d):
        try:
            color_profile = color_frame.get_stream_profile()
            depth_profile = depth_frame.get_stream_profile()

            # 🔧 SDK 내장 캘리브레이션 파라미터 활용
            print("=== SDK 내장 캘리브레이션 정보 ===")
            
            # 1) SDK에서 내부 파라미터 직접 가져오기
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
            color_distortion = color_profile.as_video_stream_profile().get_distortion()
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
            depth_distortion = depth_profile.as_video_stream_profile().get_distortion()

            print(f"Color 카메라 내부 파라미터:")
            print(f"  - fx: {color_intrinsics.fx:.2f}, fy: {color_intrinsics.fy:.2f}")
            print(f"  - cx: {color_intrinsics.cx:.2f}, cy: {color_intrinsics.cy:.2f}")
            print(f"  - 왜곡계수: k1={color_distortion.k1:.6f}, k2={color_distortion.k2:.6f}")
            
            print(f"Depth 카메라 내부 파라미터:")
            print(f"  - fx: {depth_intrinsics.fx:.2f}, fy: {depth_intrinsics.fy:.2f}")
            print(f"  - cx: {depth_intrinsics.cx:.2f}, cy: {depth_intrinsics.cy:.2f}")
            
            # 2) SDK에서 Depth-to-Color extrinsic 가져오기
            extrinsic_d2c = depth_profile.get_extrinsic_to(color_profile)
            R_d2c = np.array(extrinsic_d2c.rot, dtype=np.float32).reshape(3, 3)
            t_d2c = np.array(extrinsic_d2c.transform, dtype=np.float32)

            print(f"SDK Depth -> Color 변환:")
            print(f"  - 회전행렬:\n{R_d2c}")
            print(f"  - 평행이동: {t_d2c} mm")
            
            # 회전행렬 determinant 확인 (1이어야 정상)
            det_R_d2c = np.linalg.det(R_d2c)
            print(f"  - 회전행렬 determinant: {det_R_d2c:.6f}")

            # 3) 체커보드 3D 좌표 생성 (원점 위치 고려)
            objp = self.create_checkerboard_3d_points()
            
            print(f"\n=== 체커보드 설정 ===")
            print(f"체커보드 3D 좌표 범위:")
            print(f"- X: {objp[:, 0].min():.1f} ~ {objp[:, 0].max():.1f} mm")
            print(f"- Y: {objp[:, 1].min():.1f} ~ {objp[:, 1].max():.1f} mm")
            print(f"- Z: {objp[:, 2].min():.1f} ~ {objp[:, 2].max():.1f} mm")

            # 4) 카메라 매트릭스 구성 - Color 카메라 기준
            camera_matrix = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.cx],
                [0, color_intrinsics.fy, color_intrinsics.cy],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array([
                color_distortion.k1, color_distortion.k2,
                color_distortion.p1, color_distortion.p2,
                color_distortion.k3
            ], dtype=np.float32)

            # 5) solvePnP로 Color카메라->체커보드 변환 구하기 (정밀도 향상)
            corners_2d = corners_2d.reshape(-1, 2).astype(np.float32)
            
            print(f"\n=== PnP 해법 계산 ===")
            print(f"검출된 코너 수: {len(corners_2d)}")
            print(f"체커보드 점 수: {len(objp)}")
            
            # 🔧 다양한 PnP 방법 시도 (정확도 향상)
            pnp_methods = [
                (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
                (cv2.SOLVEPNP_EPNP, "EPNP"), 
                (cv2.SOLVEPNP_P3P, "P3P") if len(corners_2d) >= 4 else None,
                (cv2.SOLVEPNP_SQPNP, "SQPNP") if len(corners_2d) >= 3 else None
            ]
            pnp_methods = [m for m in pnp_methods if m is not None]
            
            best_error = float('inf')
            best_result = None
            
            for method, name in pnp_methods:
                try:
                    success, rvec, tvec = cv2.solvePnP(objp, corners_2d, camera_matrix, dist_coeffs, flags=method)
                    
                    if success:
                        # 재투영 오차 계산
                        reprojection_error = self.calculate_reprojection_error(objp, corners_2d, camera_matrix, dist_coeffs, rvec, tvec)
                        print(f"  - {name}: 성공, 재투영 오차: {reprojection_error:.3f}px")
                        
                        if reprojection_error < best_error:
                            best_error = reprojection_error
                            best_result = (rvec, tvec, name)
                    else:
                        print(f"  - {name}: 실패")
                        
                except Exception as e:
                    print(f"  - {name}: 오류 ({e})")
            
            if best_result is None:
                print("✗ 모든 PnP 방법이 실패했습니다")
                return False
                
            rvec, tvec, best_method = best_result
            print(f"✓ 최적 방법: {best_method} (오차: {best_error:.3f}px)")

            # 6) 🔧 solvePnP 결과 올바른 해석
            # solvePnP 반환: X_color = R_pnp * X_board + t_pnp (Board → Color)
            # 우리가 필요: X_board = R_c2b * X_color + t_c2b (Color → Board)
            # 역변환: R_c2b = R_pnp^T, t_c2b = -R_pnp^T * t_pnp
            
            R_board_to_color, _ = cv2.Rodrigues(rvec)  # solvePnP 결과: Board → Color
            t_board_to_color = tvec.flatten()
            
            # 역변환으로 Color → Board 구하기
            R_color_to_board = R_board_to_color.T
            t_color_to_board = -R_color_to_board @ t_board_to_color

            print(f"\n=== solvePnP 결과 해석 ===")
            print(f"Board → Color (solvePnP 직접 결과):")
            print(f"  - 회전: {np.degrees(rvec.flatten())} degrees")
            print(f"  - 평행이동: {t_board_to_color} mm")
            print(f"Color → Board (역변환):")
            print(f"  - 평행이동: {t_color_to_board} mm")

            # 7) 최종 변환: Depth → Board = (Color → Board) ∘ (Depth → Color)
            # T_d2b = T_c2b @ T_d2c
            R_d2b = R_color_to_board @ R_d2c
            t_d2b = R_color_to_board @ t_d2c + t_color_to_board

            print(f"\n=== 최종 변환 결과 ===")
            
            # 회전각 분석 (참고용)
            sy = np.sqrt(R_d2b[0,0]**2 + R_d2b[1,0]**2)
            roll_current = np.degrees(np.arctan2(R_d2b[2,1], R_d2b[2,2]))
            pitch_current = np.degrees(np.arctan2(-R_d2b[2,0], sy))
            yaw_current = np.degrees(np.arctan2(R_d2b[1,0], R_d2b[0,0]))
            
            print(f"Depth → Board 회전각:")
            print(f"  - Roll (X축): {roll_current:.1f}°")
            print(f"  - Pitch (Y축): {pitch_current:.1f}°") 
            print(f"  - Yaw (Z축): {yaw_current:.1f}°")
            
            # 좌표계 검증
            board_origin_in_camera = -R_d2b.T @ t_d2b
            camera_origin_in_board = t_d2b
            
            print(f"좌표계 검증:")
            print(f"  - 보드 원점 (카메라 좌표): {board_origin_in_camera}")
            print(f"  - 카메라 원점 (보드 좌표): {camera_origin_in_board}")
            print(f"  - 보드가 카메라 아래쪽에 있나? Z_board < Z_camera: {board_origin_in_camera[2] < 0}")

            # 변환 매트릭스 생성
            T_d2b = np.eye(4, dtype=np.float32)
            T_d2b[:3, :3] = R_d2b
            T_d2b[:3, 3] = t_d2b

            print(f"\n최종 Depth -> 체커보드 변환 매트릭스:")
            # 과학적 표기법 대신 소수점 표기법으로 출력
            np.set_printoptions(suppress=True, precision=6)
            print(T_d2b)
            np.set_printoptions()  # 기본 설정으로 복원

            # 캘리브레이션 결과 저장
            success_save = self.save_calibration_results(T_d2b, objp, corners_2d, 
                                                       camera_matrix, dist_coeffs, rvec, tvec)
            if not success_save:
                print("⚠ 캘리브레이션 결과 저장 실패")

            # 검증: 역변환 테스트
            T_b2d = np.linalg.inv(T_d2b)
            identity_test = T_d2b @ T_b2d
            print(f"\n변환 매트릭스 검증 (단위행렬과의 차이):")
            print(f"최대 오차: {np.max(np.abs(identity_test - np.eye(4))):.6f}")

            return True

        except Exception as e:
            print(f"✗ 캘리브레이션 계산 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_calibration_results(self, T_d2b, objp, corners_2d, camera_matrix, dist_coeffs, rvec, tvec):
        """캘리브레이션 결과를 여러 형태로 저장"""
        import json
        import datetime
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 기본 변환 행렬 파일 (다른 스크립트들이 사용)
            np.savetxt("depth_to_checkerboard_transform.txt", T_d2b, fmt='%.6f')
            print(f"✓ 변환 매트릭스 저장: depth_to_checkerboard_transform.txt")
            
            # 2. 타임스탬프가 포함된 백업 파일
            backup_file = f"calibration_backup_{timestamp}.txt"
            np.savetxt(backup_file, T_d2b, fmt='%.6f')
            print(f"✓ 백업 파일 저장: {backup_file}")
            
            # 3. 상세 캘리브레이션 정보 JSON 파일
            calib_info = {
                "timestamp": timestamp,
                "checkerboard_size": self.checkerboard_size,
                "square_size_mm": self.square_size,
                "origin_corner": self.origin_corner,
                "transform_matrix": T_d2b.tolist(),
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coeffs": dist_coeffs.tolist(),
                "rotation_vector": rvec.flatten().tolist(),
                "translation_vector": tvec.flatten().tolist(),
                "num_corners_detected": len(corners_2d),
                "reprojection_error_px": self.calculate_reprojection_error(
                    objp, corners_2d, camera_matrix, dist_coeffs, rvec, tvec
                )
            }
            
            info_file = f"calibration_info_{timestamp}.json"
            with open(info_file, 'w') as f:
                json.dump(calib_info, f, indent=2)
            print(f"✓ 상세 정보 저장: {info_file}")
            
            # 4. 최신 캘리브레이션 정보 (항상 덮어쓰기)
            with open("latest_calibration_info.json", 'w') as f:
                json.dump(calib_info, f, indent=2)
            print(f"✓ 최신 정보 저장: latest_calibration_info.json")
            
            # 5. 캘리브레이션 로그
            log_entry = f"\n[{timestamp}] 캘리브레이션 완료\n"
            log_entry += f"  - 체커보드: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 교차점\n"
            log_entry += f"  - 사각형 크기: {self.square_size}mm\n"
            log_entry += f"  - 원점: {self.origin_corner}\n"
            log_entry += f"  - 검출된 코너: {len(corners_2d)}개\n"
            log_entry += f"  - 재투영 오차: {calib_info['reprojection_error_px']:.3f}px\n"
            log_entry += f"  - 변환 행렬:\n"
            for row in T_d2b:
                log_entry += f"    [{row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f} {row[3]:9.6f}]\n"
            
            with open("calibration_log.txt", 'a') as f:
                f.write(log_entry)
            print(f"✓ 로그 추가: calibration_log.txt")
            
            return True
            
        except Exception as e:
            print(f"✗ 캘리브레이션 결과 저장 오류: {e}")
            return False
    
    def calculate_reprojection_error(self, objp, corners_2d, camera_matrix, dist_coeffs, rvec, tvec):
        """재투영 오차 계산"""
        try:
            proj_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            proj_points = proj_points.reshape(-1, 2)
            error = np.mean(np.linalg.norm(proj_points - corners_2d.reshape(-1, 2), axis=1))
            return float(error)
        except:
            return 0.0
    
    def cleanup(self):
        if self.pipeline:
            self.pipeline.stop()

def main():
    calib = FinalDepthCalibration()
    
    try:
        if not calib.setup_camera():
            return
            
        calib.capture_checkerboard()
    except KeyboardInterrupt:
        pass
    finally:
        calib.cleanup()

if __name__ == "__main__":
    main()