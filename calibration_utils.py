#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캘리브레이션 공통 유틸리티 함수들
"""

import numpy as np
import json
import os
import glob
from datetime import datetime

def load_latest_calibration():
    """가장 최신 캘리브레이션 결과를 로드 (변환 행렬만 반환)"""
    
    # 1. 우선순위: latest_calibration_info.json 파일
    if os.path.exists("latest_calibration_info.json"):
        try:
            with open("latest_calibration_info.json", 'r') as f:
                info = json.load(f)
            
            transform_matrix = np.array(info["transform_matrix"], dtype=np.float32)
            print(f"✓ 최신 캘리브레이션 로드: {info['timestamp']}")
            print(f"  - 체커보드: {info['checkerboard_size'][0]}x{info['checkerboard_size'][1]} 교차점")
            print(f"  - 사각형 크기: {info['square_size_mm']}mm")
            print(f"  - 원점: {info['origin_corner']}")
            print(f"  - 재투영 오차: {info.get('reprojection_error_px', 'N/A')}px")
            
            # 변환 행렬을 실수 표기법으로 출력
            print("  - 변환 행렬:")
            np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
            for i, row in enumerate(transform_matrix):
                print(f"    [{row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f} {row[3]:9.6f}]")
            np.set_printoptions()  # 기본 설정으로 복원
            
            return transform_matrix
            
        except Exception as e:
            print(f"⚠ latest_calibration_info.json 읽기 실패: {e}")
    
    # 2. 대안: depth_to_checkerboard_transform.txt 파일
    if os.path.exists("depth_to_checkerboard_transform.txt"):
        try:
            transform_matrix = np.loadtxt("depth_to_checkerboard_transform.txt", dtype=np.float32)
            print("✓ 변환행렬 로드: depth_to_checkerboard_transform.txt")
            print(f"  - 원점: [{transform_matrix[0,3]:.1f}, {transform_matrix[1,3]:.1f}, {transform_matrix[2,3]:.1f}]mm")
            
            # 변환 행렬을 실수 표기법으로 출력
            print("  - 변환 행렬:")
            np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
            for i, row in enumerate(transform_matrix):
                print(f"    [{row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f} {row[3]:9.6f}]")
            np.set_printoptions()  # 기본 설정으로 복원
            
            return transform_matrix
            
        except Exception as e:
            print(f"⚠ depth_to_checkerboard_transform.txt 읽기 실패: {e}")
    
    # 3. 최후 수단: 가장 최근 백업 파일 찾기
    backup_files = glob.glob("calibration_backup_*.txt")
    if backup_files:
        # 파일명에서 타임스탬프 추출하여 정렬
        backup_files.sort(reverse=True)  # 최신 파일이 먼저
        latest_backup = backup_files[0]
        
        try:
            transform_matrix = np.loadtxt(latest_backup, dtype=np.float32)
            print(f"✓ 백업에서 로드: {latest_backup}")
            
            return transform_matrix
            
        except Exception as e:
            print(f"⚠ 백업 파일 읽기 실패: {e}")
    
    # 4. 모든 방법 실패시 None 반환
    print("✗ 캘리브레이션 파일을 찾을 수 없습니다.")
    print("  다음 파일 중 하나가 필요합니다:")
    print("  - latest_calibration_info.json")
    print("  - depth_to_checkerboard_transform.txt")
    print("  - calibration_backup_YYYYMMDD_HHMMSS.txt")
    
    return None

def get_fallback_transform():
    """기본 변환 행렬 (하드코딩)"""
    print("⚠ 기본 변환 행렬 사용 (하드코딩된 값)")
    return np.array([
        [0.994144,    0.001478,    0.026562, -280.04895 ],
        [-0.016698,    0.808585,    0.588139, -114.991554],
        [-0.020528,   -0.588378,    0.808323,  656.5049],
        [0.000000,   0.000000,   0.000000,   1.000000]
    ], dtype=np.float32)

def load_transform_matrix_with_fallback():
    """변환 행렬 로드 (실패시 기본값 사용)"""
    transform_matrix = load_latest_calibration()
    
    if transform_matrix is None:
        transform_matrix = get_fallback_transform()
    
    return transform_matrix

def print_calibration_summary():
    """캘리브레이션 결과 요약 출력"""
    transform_matrix = load_latest_calibration()
    
    if transform_matrix is None:
        print("캘리브레이션 파일이 없습니다. 먼저 캘리브레이션을 수행하세요.")
        return
    
    # 추가 정보는 JSON에서 로드
    info = None
    if os.path.exists("latest_calibration_info.json"):
        try:
            with open("latest_calibration_info.json", 'r') as f:
                info = json.load(f)
        except:
            pass
    
    print("\n" + "="*50)
    print("캘리브레이션 결과 요약")
    print("="*50)
    
    if info:
        print(f"타임스탬프: {info['timestamp']}")
        print(f"체커보드: {info['checkerboard_size'][0]}x{info['checkerboard_size'][1]} 교차점")
        print(f"사각형 크기: {info['square_size_mm']}mm")
        print(f"원점 위치: {info['origin_corner']}")
        print(f"재투영 오차: {info.get('reprojection_error_px', 'N/A')}")
    
    print("\n변환 행렬:")
    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')
    for i, row in enumerate(transform_matrix):
        print(f"  [{row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f} {row[3]:9.6f}]")
    np.set_printoptions()  # 기본 설정으로 복원
    
    print(f"\n원점 위치: [{transform_matrix[0,3]:.1f}, {transform_matrix[1,3]:.1f}, {transform_matrix[2,3]:.1f}]mm")
    print("="*50)

if __name__ == "__main__":
    # 테스트용
    print_calibration_summary()
