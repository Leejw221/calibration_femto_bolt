#!/bin/bash

# Femto Bolt 캘리브레이션 및 실시간 시각화 스크립트
# 작성자: GitHub Copilot
# 날짜: 2025-08-11

echo "======================================"
echo "Femto Bolt 캘리브레이션 및 시각화"
echo "======================================"

# 작업 디렉토리로 이동
cd /home/leejungwook/Femto_bolt_calibration

echo ""
echo "1️⃣ 체커보드 캘리브레이션 시작..."
echo "   - 9x6 교차점, 55mm 사각형"
echo "   - bottom_right 원점"
echo "   - 카메라 25° 기울기 고려"
echo ""

# 캘리브레이션 실행
python3 calibration_by_sdk.py

# 캘리브레이션 성공 여부 확인
if [ $? -eq 0 ] && [ -f "depth_to_checkerboard_transform.txt" ]; then
    echo ""
    echo "✅ 캘리브레이션 완료!"
    echo "   변환 행렬: depth_to_checkerboard_transform.txt"
    echo "   상세 정보: latest_calibration_info.json"
    echo ""
    
    # 캘리브레이션 결과 요약 출력
    if [ -f "latest_calibration_info.json" ]; then
        echo "📊 캘리브레이션 결과 요약:"
        python3 -c "
import json
try:
    with open('latest_calibration_info.json', 'r') as f:
        info = json.load(f)
    print(f'   - 체커보드: {info[\"checkerboard_size\"][0]}x{info[\"checkerboard_size\"][1]} 교차점')
    print(f'   - 사각형 크기: {info[\"square_size_mm\"]}mm')
    print(f'   - 검출된 코너: {info[\"num_corners_detected\"]}개')
    print(f'   - 재투영 오차: {info[\"reprojection_error_px\"]:.3f}px')
    print(f'   - 타임스탬프: {info[\"timestamp\"]}')
except:
    print('   - 정보 파일 읽기 실패')
"
    fi
    
    echo ""
    echo "2️⃣ 실시간 포인트 클라우드 시각화 시작..."
    echo "   - 캘리브레이션 결과 자동 적용"
    echo "   - 좌표축: 빨강(X), 초록(Y), 파랑(Z)"
    echo "   - ESC 키로 종료"
    echo ""
    
    # 잠시 대기
    sleep 2
    
    # 실시간 시각화 실행
    python3 open3d_visualization.py
    
else
    echo ""
    echo "❌ 캘리브레이션 실패!"
    echo "   다시 시도하거나 체커보드 설정을 확인하세요."
    echo ""
    exit 1
fi

echo ""
echo "🎉 모든 작업 완료!"
echo "======================================"
