import streamlit as st
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from video_handler import VideoHandler
import av
import torch
import cv2
from lane_detection import LaneLines  # Import lane detection class

# WebRTC Configuration for video streaming
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize the video handler (model for YOLOv8)
        self.video_handler = VideoHandler()
        self.lane_detector = LaneLines()  # Initialize lane detector

    def recv(self, frame):
        # Convert the frame into an OpenCV image (numpy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame for object tracking with YOLOv8
        processed_frame, tracking_results = self.video_handler.process_frame(img)
        
        # Process the frame for lane detection
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for lane detection
        lane_frame = self.lane_detector.forward(gray_frame)  # Perform lane detection
        
        # Overlay lane lines on the processed frame
        final_frame = cv2.addWeighted(processed_frame, 1, lane_frame, 0.7, 0)

        # Optional: Add tracking results to Streamlit interface
        st.session_state.tracking_results = tracking_results
        
        # Return the processed frame
        return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

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
