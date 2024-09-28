import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from video_handler import VideoHandler
import av
import cv2

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_handler = VideoHandler()
        self.danger_alarm = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the frame for object (pedestrian) and lane detection
        processed_frame, tracking_results, danger_alarm = self.video_handler.process_frame(img)

        # Update danger alarm state
        self.danger_alarm = danger_alarm

        # Display danger alarm within the frame (if any)
        if self.danger_alarm:
            cv2.putText(processed_frame, "DANGER! Pedestrian in Lane!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Return the processed frame for streaming
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.title("Real-Time Pedestrian and Lane Detection with Danger Alarm")

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Show warning or information based on the detected danger
    if "danger_alarm" in st.session_state and st.session_state.danger_alarm:
        st.error("Danger! Pedestrian detected in the vehicle lane.")
    else:
        st.info("System is monitoring. No danger detected.")

if __name__ == "__main__":
    main()
