import cv2
import torchvision
import os
def split_and_resize_video(input_video_path, output_dir, output_filenames, target_height=320,target_width=1024, fps=10,skip=False):
    left_savepath = os.path.join(output_dir, f"{output_filenames[0]}.mp4")
    right_savepath = os.path.join(output_dir, f"{output_filenames[1]}.mp4")
    if skip and  os.path.exists(left_savepath) and os.path.exists(right_savepath):
        print("skipping",output_filenames[0])
        return 
    # Load video
    video, _, info = torchvision.io.read_video(input_video_path)
    video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
    
    # Check video dimensions
    T, C, H, W = video.shape
    assert W == 1030, f"Expected video width of 1030, but got {W}"
    assert H == 324, f"Expected video height of 324, but got {H}"
    assert len(output_filenames) == 2, "Output filenames list should contain exactly 2 filenames."

    # Resize height to target_height (320)
    video = torchvision.transforms.functional.center_crop(video, [target_height, target_width])  # Crop to (T, C, 320, 1024)
    
    # Split video into two parts
    left_video = video[:, :, :, :512]  # (T, C, 320, 512)
    right_video = video[:, :, :, -512:]  # (T, C, 320, 512)
    
    # Save the left video
    if skip and not os.path.exists(left_savepath):
        left_video = left_video.permute(0, 2, 3, 1)  # Convert to (T, H, W, C)
        torchvision.io.write_video(
            left_savepath, left_video, fps=fps, video_codec="h264", options={"crf": "10"}
        )
    if skip and not os.path.exists(right_savepath):
        # Save the right video
        right_video = right_video.permute(0, 2, 3, 1)  # Convert to (T, H, W, C)
        torchvision.io.write_video(
            right_savepath, right_video, fps=fps, video_codec="h264", options={"crf": "10"}
        )
