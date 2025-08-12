import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from calibration_utils import load_latest_calibration

# 최신 캘리브레이션 결과 자동 로드
try:
    camera_to_base = load_latest_calibration()
    if camera_to_base is None:
        raise ValueError("캘리브레이션 결과가 None입니다.")
    
    # 행렬이 올바른 형태인지 확인
    camera_to_base = np.array(camera_to_base, dtype=np.float64)
    if camera_to_base.shape != (4, 4):
        raise ValueError(f"변환 행렬의 형태가 잘못되었습니다: {camera_to_base.shape}")
    
    # camera_to_base = np.array([
    #                 [  0.007131,  -0.91491,    0.403594,  51.161087],
    #                 [ -0.994138,   0.003833,   0.02656,  -9.179153],
    #                 [ -0.025717,  -0.403641,  -0.914552, 508.20871 ],
    #                 [  0.,         0. ,        0. ,        1.      ]],
    #                 dtype=np.float64)

    print("✅ 최신 캘리브레이션 결과를 로드했습니다.")
    print(f"변환 행렬:\n{camera_to_base}")
    
except Exception as e:
    print(f"⚠️ 캘리브레이션 로드 실패: {e}")
    print("기본 변환 행렬을 사용합니다.")
    camera_to_base = np.array([
                    [ 0.0,        -0.9063,      0.4226,    110.],
                    [ -1.0,        0.,          0.,          0.],
                    [0.0,          -0.4226,      -0.9063,     510.       ],
                    [ 0.,          0.,          0.,          1.         ]
                ], dtype=np.float64)


def main():
    print("🚀 Femto Bolt 실시간 포인트 클라우드 시작...")
    print("   - 창을 닫거나 Ctrl+C로 종료")
    print("   - 마우스로 시점 조작 가능")
    print("")

    pipeline = ob.Pipeline()
    cfg = ob.Config()
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
    cfg.enable_stream(depth_profile)
    cfg.enable_stream(color_profile)

    pipeline.enable_frame_sync()
    pipeline.start(cfg)

    align = ob.AlignFilter(align_to_stream = ob.OBStreamType.DEPTH_STREAM)
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)

    use_vis = True
    if use_vis:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Femto Bolt - Calibrated Point Cloud', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        pcd = o3d.geometry.PointCloud()
    
    first_iter = True
    
    t_cnt=0
    try:
        while True:
            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc=np.asarray(point_cloud) 
            pc = pc[pc[:, 2] > 0.0]

            if use_vis:
                # Homogeneous coordinates
                points = pc[:, :3]
                points_h = np.hstack([points, np.ones((points.shape[0], 1))])
                
                # Transform
                transformed_points_h = (camera_to_base @ points_h.T).T
                
                # Back to 3D
                transformed_points = transformed_points_h[:, :3]
                
                pcd.points = o3d.utility.Vector3dVector(transformed_points)
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6] / 255.0)
                if first_iter:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    # Add base coordinate system at the origin
                    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=150, origin=[0, 0, 0])
                    vis.add_geometry(base_frame)

                    # Add camera coordinate system, transformed to its pose relative to the base
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=75, origin=[0, 0, 0])  # Smaller size to distinguish
                    camera_frame.transform(camera_to_base)
                    vis.add_geometry(camera_frame)
                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([-1.0, 0.0, 0.0])
                    ctr.set_up([0.0, 0.0, 1.0])
                    ctr.set_zoom(0.4)
                    first_iter = False
                
                vis.update_geometry(pcd)
                
                # 윈도우 상태 확인 및 이벤트 처리
                if not vis.poll_events():
                    print("시각화 창이 닫혔습니다.")
                    break
                    
                vis.update_renderer()

    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        pipeline.stop()
        if 'vis' in locals():
            vis.destroy_window()

if __name__ == '__main__':
    main()
