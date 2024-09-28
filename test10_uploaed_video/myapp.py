import streamlit as st
from video_handler import VideoHandler

def main():
    st.title("Object and Lane Detection on Uploaded Video")

    with st.sidebar:
        st.title("Configuration")
        uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4"])
        track_bar = st.sidebar.progress(0.0, text="Progressbar for tracking")

    if uploaded_video is not None:
        my_video = VideoHandler(uploaded_video)

        # Display the original uploaded video
        st.video(uploaded_video)

        fps, width, height, total_frames = my_video.get_video_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Frames per sec.", value=f"{fps}")
        col2.metric(label="Width", value=f"{width} px")
        col3.metric(label="Height", value=f"{height} px")
        col4.metric(label="Frames", value=f"{total_frames}")

        def update_progressbar(frame_counter):
            track_bar.progress(frame_counter / total_frames)

        # Process the video
        _, processed_video_path = my_video.track(progressbar_callback=update_progressbar)

        # Ensure the video is fully processed before displaying
        st.video(processed_video_path)  # Display the processed video with object and lane detection results
        del my_video

if __name__ == "__main__":
    main()
