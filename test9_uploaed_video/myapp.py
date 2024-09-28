import streamlit as st
import cv2
import tempfile
import os
from video_handler import VideoHandler

def main():
    st.title("Object and Lane Detection on Uploaded Video")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Initialize the video handler (object and lane detection)
        video_handler = VideoHandler()

        # Process the uploaded video
        process_uploaded_video(temp_file.name, video_handler)

        # Remove the temporary file after processing
        os.remove(temp_file.name)

# def process_uploaded_video(video_path, video_handler):
#     # Read the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Create a temporary file for the output video
#     output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     out = cv2.VideoWriter(output_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#     # Process each frame
#     progress_bar = st.progress(0)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     for i in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform object and lane detection
#         processed_frame, _ = video_handler.process_frame(frame)

#         # Write the processed frame to the output video
#         out.write(processed_frame)

#         # Update the progress bar
#         progress_bar.progress((i + 1) / frame_count)

#     # Release video resources
#     cap.release()
#     out.release()

#     st.success("Video processing completed!")

#     # Display the processed video
#     st.video(output_video.name)

def process_uploaded_video(video_path, video_handler):
    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a temporary file for the output video
    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # Ensure that the codec is compatible (H.264 codec with mp4 container)
    out = cv2.VideoWriter(output_video.name, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

    # Process each frame
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object and lane detection
        processed_frame, _ = video_handler.process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Update the progress bar
        progress_bar.progress((i + 1) / frame_count)

    # Release video resources
    cap.release()
    out.release()

    st.success("Video processing completed!")

    # Display the processed video
    st.video(output_video.name)


if __name__ == "__main__":
    main()
