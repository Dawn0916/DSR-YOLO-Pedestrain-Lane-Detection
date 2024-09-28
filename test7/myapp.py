import streamlit as st
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from video_handler import VideoHandler
import av
import cv2
import numpy as np

# WebRTC Configuration for video streaming
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_interval = 10  # Perform detection every 5 frames
        # Initialize the video handler (model for YOLOv8 and Lane detection)
        self.video_handler = VideoHandler()

    def recv(self, frame):
        self.frame_count += 1
        # Convert the frame into an OpenCV image (numpy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Perform object detection and lane detection on every 5th frame
        if self.frame_count % self.detection_interval == 0:
            # Process the frame for object detection and lane detection
            combined_frame, tracking_results = self.video_handler.process_frame(img)
        else:
            # Skip detection and just display the image for performance
            combined_frame = img
        
        # Optional: Add tracking results to Streamlit interface
        st.session_state.tracking_results = tracking_results

        # Return the processed frame
        return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")

def main():
    st.title("Real-Time Object and Lane Detection with YOLOv8")

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if "tracking_results" in st.session_state:
        tracking_results = st.session_state.tracking_results
        if tracking_results is not None and not tracking_results.empty:
            # Display tracking results
            st.write(tracking_results)

            # Plot class counts over time using Plotly
            group_data = tracking_results[["frame_no", "class_name"]].groupby(
                ["frame_no", "class_name"]).size().reset_index(name="count")
            final_df = group_data.pivot(index="frame_no", columns="class_name", values="count").fillna(0)

            fig = px.line(final_df,
                          x=final_df.index,
                          y=tracking_results.class_name.unique(),
                          title='Class counts for each frame')

            fig.update_xaxes(title_text='Frame number')
            fig.update_yaxes(title_text='Counts')

            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
